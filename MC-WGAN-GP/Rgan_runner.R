# name of script you want to train
script <- ' gan_trainer_multi_v1.py'

# Do you want to save the parameters? 
save_params <- TRUE
param_list <- list(loss_penalty = c(1,5,10,20),
                   gen_bn_decay = c(0,.1,.25,.5,.9),
                   gen_l2_regularization = c(0,.001),
                   disc_l2_regularization = c(0,.001),
                   mini_batch = c("'True'","'False'"))
runs <- expand.grid(param_list)

# How many total runs? 
n_run <- nrow(runs)

# how would you describe these runs?  
description <- "multi_gan"

run_description <- paste0(format(Sys.time(), "%Y%m%d"),"_",
                          description)


# Here is where you can tune the GAN. Make sure that the total number of options
# for each parameter is equal to n_run. Also make sure strings are strings within
# stings, this is weird but it is how the parser works. 

sim_num <- 1:n_run
num_epochs <- rep(x = 10000, n_run) # This is large just so the GAN will train through the whole week
disc_epochs <- rep(x = 2, n_run)
gen_epochs <- rep(x = 1, n_run)

batch_size <- rep(x = 500, n_run)
z_size <- rep(55, n_run)


generator_hidden_sizes <- rep(x = "'100,100,100'", n_run)
discriminator_hidden_sizes <- rep(x = "'100'", n_run)


loss_penalty <- runs$loss_penalty
gen_bn_decay <- runs$gen_bn_decay
mini_batch <- runs$mini_batch
gen_l2_regularization <- runs$gen_l2_regularization
disc_l2_regularization <- runs$disc_l2_regularization

data_size <- rep(x = 55, n_run) # number of columns within the dummied data
var_locs <- rep(x = "'0,1'", n_run) # Location of continuous variables (string)
variable_sizes <- rep(x = "'5,12,7,2,10,12,5'", n_run) # Size of each categorical variable (string)
disc_leaky_param <- rep(x = 0.2, n_run)
gen_learning_rate <- rep(x = 0.001, n_run)
disc_learning_rate <- rep(x = 0.001, n_run)
disc_bn_decay <- rep(x = 0.0, n_run) # should stay at 0 if critic == True
critic <- rep(x = "'True'", n_run)  # if false then disc outputs between 0 and 1

# This in only needed for the single GAN, the value doesn't matter if you are training the multi GAN
auto_encoder_loc <- rep(x = "'./saved_parameters/autoencoder_190220'", n_run)


# First Priority
# - gen_bn_decay
# - loss_penalty
# - l2_reg

# Second Priority
# - mini_batch
# - hidden sizes


run_info <- apply(runs,1,function(x) paste0(gsub("\\.","",gsub("'","",x)), collapse = '_'))
run_info <- trimws(run_info)

## The next couple of lines creates the locations to save paramters. 

param_save_loc <- paste0("./training_iterations/", run_description)
disc_save_loc <- paste0("'",param_save_loc,'/disc/disc_',1:n_run,"_",run_info,"'")
gen_save_loc <- paste0("'",param_save_loc,'/gen/gen_',1:n_run,"_",run_info,"'")
disc_optim_save_loc <- paste0("'",param_save_loc,'/disc_optim/disc_optim_',1:n_run,"_",run_info,"'")
gen_optim_save_loc <- paste0("'",param_save_loc,'/disc_optim/dis_optim_',1:n_run,"_",run_info,"'")
fig_save_path <- paste0("'",param_save_loc,'/figs/fig_',1:n_run,run_info,"'")
out_loc <- paste0("./output_files/",run_description,1:n_run,run_info)

# create folder for output files
dir.create("./output_files")


# if you choose to save parameters this creates the needed directories
if(save_params){
  dir.create(file.path("./training_iterations/"))
  dir.create(file.path(param_save_loc))
  dir.create(path = file.path(paste0(param_save_loc,'/disc')))
  dir.create(path = file.path(paste0(param_save_loc,'/gen')))
  dir.create(path = file.path(paste0(param_save_loc,'/disc_optim')))
  dir.create(path = file.path(paste0(param_save_loc,'/gen_optim')))
  dir.create(path = file.path(paste0(param_save_loc,'/figs')))
  save_bool <- rep("'True'", n_run)
} else {
  save_bool <- rep("'False'", n_run)
}


sims <- data.frame(sim_num, num_epochs, batch_size, disc_epochs, gen_epochs, loss_penalty, disc_bn_decay, gen_bn_decay, mini_batch, disc_leaky_param,          
           critic, gen_l2_regularization, disc_l2_regularization, gen_learning_rate, disc_learning_rate, data_size, z_size, var_locs, variable_sizes,
           generator_hidden_sizes, discriminator_hidden_sizes, auto_encoder_loc, disc_save_loc, gen_save_loc, disc_optim_save_loc,        
           gen_optim_save_loc, fig_save_path, out_loc, save_bool, stringsAsFactors = FALSE)

# This creates the script to run
create_runner_script <- function(text){
  (paste0(' python3',
          script,' --num_epochs ',text['num_epochs'], ' --batch_size ',text['batch_size'],' --disc_epochs ',text['disc_epochs'], ' --gen_epochs ',text['gen_epochs'],' --z_size ',text['z_size'], ' --data_size ',text['data_size'],' --disc_bn_decay ',text['disc_bn_decay'],' --disc_leaky_param ',text['disc_leaky_param'],' --mini_batch ',text['mini_batch'],' --critic ',text['critic'],' --gen_bn_decay ',text['gen_bn_decay'],' --gen_l2_regularization ',text['gen_l2_regularization'],' --disc_l2_regularization ',text['disc_l2_regularization'],' --gen_learning_rate ',text['gen_learning_rate'], ' --disc_learning_rate ',text['disc_learning_rate'],' --loss_penalty ',text['loss_penalty'], ' --var_locs ',text['var_locs'],' --variable_sizes ',text['variable_sizes'],' --generator_hidden_sizes ',text['generator_hidden_sizes'],' --discriminator_hidden_sizes ',text['discriminator_hidden_sizes'],' --auto_encoder_loc ',text['auto_encoder_loc'], ' --disc_save_loc ',text['disc_save_loc'], ' --gen_save_loc ',text['gen_save_loc'], ' --disc_optim_save_loc ',text['disc_optim_save_loc'],' --gen_optim_save_loc ',text['gen_optim_save_loc'],' --save_bool ',text['save_bool'],' --fig_save_path ',text['fig_save_path'], ' > ',text['out_loc'],' 2>&1 & '))
}


# This executes the scripts
for(i in 1:10){
  test1 <- create_runner_script(sims[i,])
  system(test1)
}


