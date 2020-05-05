library(cellar)
library(ctgan)
library(tidyverse)
library(rsample)

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

# synthesis
synthesizer <- ctgan()
train_syn <- modeling_data %>% 
  mutate_at("num_claims", as.character)
synthesizer %>%
  fit(train_syn,batch_size = 10000, epochs = 300)

synthesizer %>%
  ctgan_save("~/ctgan_results/claims_ctgan")

syn <- synthesizer %>%
ctgan_sample(n = nrow(modeling_data), batch_size = 10000) %>% 
  mutate(num_claims = as.integer(num_claims),
         exposure = pmin(pmax(1/365, exposure), 1))

write_csv(syn, "~/ctgan_results/claims_gen_data.csv")
# ctgan_load("~/ctgan_results/claims_ctgan")