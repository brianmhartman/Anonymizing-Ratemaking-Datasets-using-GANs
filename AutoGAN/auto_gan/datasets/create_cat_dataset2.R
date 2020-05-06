library(CASdatasets)
library(tidyverse)
data("freMTPLfreq")
tab1 <- table(as.factor(freMTPLfreq$Exposure))
str(tab1)
plot(as.vector(tab1))


exposure_vec <- numeric(nrow(freMTPLfreq))

exposure_vec <- exposure_vec + (freMTPLfreq$Exposure < .1) * 1
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .1 & freMTPLfreq$Exposure <.2) * 2
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .2 & freMTPLfreq$Exposure <.3) * 3
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .3 & freMTPLfreq$Exposure <.4) * 4
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .4 & freMTPLfreq$Exposure <.5) * 5
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .5 & freMTPLfreq$Exposure <.6) * 6
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .6 & freMTPLfreq$Exposure <.7) * 7
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .7 & freMTPLfreq$Exposure <.8) * 8
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .8 & freMTPLfreq$Exposure <.9) * 9
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure >= .9 & freMTPLfreq$Exposure < 1) * 10
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure == 1) * 11
exposure_vec <- exposure_vec + (freMTPLfreq$Exposure > 1 ) * 12

table(exposure_vec)

summary(freMTPLfreq$Density)
plot(freMTPLfreq$Density)



# 25000, 20000, 15000
density_vec <- numeric(nrow(freMTPLfreq))
density_vec <- density_vec + (freMTPLfreq$Density < 5000) * 1
density_vec <- density_vec + (freMTPLfreq$Density >= 5000 & freMTPLfreq$Density < 10000) * 2
density_vec <- density_vec + (freMTPLfreq$Density >= 10000 & freMTPLfreq$Density < 15000) * 3
density_vec <- density_vec + (freMTPLfreq$Density >= 15000 & freMTPLfreq$Density < 25000) * 4
density_vec <- density_vec + (freMTPLfreq$Density >= 25000) * 5
table(density_vec)


car_vec <- freMTPLfreq$CarAge
car_vec[car_vec > 16 &car_vec<19] <-17
car_vec[car_vec > 18 ] <-19
table(car_vec)

plot(density(freMTPLfreq$DriverAge))
table(freMTPLfreq$DriverAge)

driver_vec <- findInterval(freMTPLfreq$DriverAge, seq(20,100, by = 5))

model_dat <- data.frame(freMTPLfreq[,c('ClaimNb','Power','Brand','Gas','Region')],
                                    'DriverAge' = driver_vec, 
                                    'Exposure_cat' = exposure_vec,
                                    'Density_cat' = density_vec,
                                    'CarAge_cat' = car_vec)

apply(model_dat,2,function(x) length(unique(x)))

sum(apply(model_dat,2,function(x) length(unique(x))))


readr::write_csv(model_dat, path = 'cat_policy_dat.csv')

table(freMTPLfreq$Power)
      