#
# model_params.py
#
# Configurable parameters for the model to be trained. 

# The more workers, the better! Don't fry your CPU, though.
# Pytorch recommends no higher than 20 workers, although that doesn't
# take into account your CPU chipset. YMMV
#
# Note that it's been observed that 20 workers results in a "bouncing"
# behavior (Ex) 190ms 1 step, 390ms the next)
data_loader_num_workers = 18

# Model Architecture
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

# Training Parameters
#learning_rate_init = 1e-4 
#learning_rate_init = 0.0009
learning_rate_init = 0.00005
model_epochs = 1000

model_similarity_weight_grad_modifier = 0.01
model_similarity_bias_grad_modifier = 0.01
model_grad_clip = 1

# Authors of the paper use N = 64 and M = 10 as parameters for the
# batch size. (Hypothetically, enrolling a speaker in speaker
# verification should only take about 10 utterances)
speakers_per_batch = 64
utterances_per_speaker = 10