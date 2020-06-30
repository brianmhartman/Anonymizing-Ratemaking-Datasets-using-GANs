# location of csv file with parameters
file_location <- 'training_iterations/iteration_records/190327_multi.csv'
# name of script you want to train
script <- ' gan_trainer_multi_v1.py'

# create folder for output files
dir.create("./output_files")

sims <- read.csv(file_location, stringsAsFactors = FALSE)
create_runner_script <- function(text){
  (paste0(' python3',
    script,' --num_epochs ',text['num_epochs'], ' --batch_size ',text['batch_size'],' --disc_epochs ',text['disc_epochs'], ' --gen_epochs ',text['gen_epochs'],' --z_size ',text['z_size'], ' --data_size ',text['data_size'],' --disc_bn_decay ',text['disc_bn_decay'],' --disc_leaky_param ',text['disc_leaky_param'],' --mini_batch ',text['mini_batch'],' --critic ',text['critic'],' --gen_bn_decay ',text['gen_bn_decay'],' --gen_l2_regularization ',text['gen_l2_regularization'],' --disc_l2_regularization ',text['disc_l2_regularization'],' --gen_learning_rate ',text['gen_learning_rate'], ' --disc_learning_rate ',text['disc_learning_rate'],' --loss_penalty ',text['loss_penalty'], ' --var_locs ',text['var_locs'],' --variable_sizes ',text['variable_sizes'],' --generator_hidden_sizes ',text['generator_hidden_sizes'],' --discriminator_hidden_sizes ',text['discriminator_hidden_sizes'],' --auto_encoder_loc ',text['auto_encoder_loc'], ' --disc_save_loc ',text['disc_save_loc'], ' --gen_save_loc ',text['gen_save_loc'], ' --disc_optim_save_loc ',text['disc_optim_save_loc'],' --gen_optim_save_loc ',text['gen_optim_save_loc'],' --fig_save_path ',text['fig_save_path'], ' > ',text['out_loc'],' 2>&1 & '))
}


for(i in 1:nrow(sims)){
  test1 <- create_runner_script(sims[i,])
  system(test1)
}


