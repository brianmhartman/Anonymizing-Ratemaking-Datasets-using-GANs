library(cellar)
library(ctgan)
library(tidyverse)
library(rsample)

set.seed(1867) # year of Canadian Independence


# download data if needed
policies <- cellar_pull("fr_tpl2_policies")

# simple transformations
modeling_data <- policies %>% 
  mutate(
    num_claims = pmin(num_claims, 4),
    exposure = pmin(exposure, 1),
    vehicle_power = ifelse(as.integer(vehicle_power) > 9, "9", vehicle_power),
    vehicle_age = case_when(
      vehicle_age < 1 ~ "< 1",
      vehicle_age <= 10 ~ "[1, 10]",
      TRUE ~ "> 10"
    ),
    driver_age = case_when(
      driver_age < 21 ~ "< 21",
      driver_age < 26 ~ "[21, 26)",
      driver_age < 31 ~ "[26, 31)",
      driver_age < 41 ~ "[41, 51)",
      driver_age < 71 ~ "[51, 71)", 
      TRUE ~ ">= 71"
    ),
    bonus_malus = pmin(bonus_malus, 150),
    density = log(density)#,
  ) %>% 
  select(-policy_id)

# separate the data
split_70 <- modeling_data %>% 
  sample_frac(0.7)
split_15 <- modeling_data %>% 
  anti_join(split_70) %>% 
  sample_frac(0.5)
holdout <- modeling_data %>% 
  anti_join(split_70) %>% 
  anti_join(split_15)

# synthesis on split_70
synthesizer_70 <- ctgan()
train_syn_70 <- split_70 %>% 
  mutate_at("num_claims", as.character)
synthesizer_70 %>%
  fit(train_syn_70, batch_size = 10000, epochs = 300)
# serialize the model
synthesizer_70 %>%
  ctgan_save("~/ctgan_results/claims_ctgan_70")

# synthesis on split_15
synthesizer_15 <- ctgan()
train_syn_15 <- split_15 %>% 
  mutate_at("num_claims", as.character)
synthesizer_15 %>%
  fit(train_syn_15, batch_size = 10000, epochs = 300)
# serialize the model
synthesizer_15 %>%
  ctgan_save("~/ctgan_results/claims_ctgan_15")

# write holdout to file for furthur study
write_csv(holdout, "~/ctgan_results/holdout_15.csv")