import numpy as np
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import sys, os
import horovod.tensorflow.keras as hvd
import utils
from architecture import Classifier

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/T2K/", help="Folder containing input files")
parser.add_option("--batch", type=int, default=512, help="Batch size")
parser.add_option("--epoch", type=int, default=200, help="Max epoch")
parser.add_option("--warm_epoch", type=int, default=5, help="Warm up epochs")
parser.add_option("--reduce_epoch", type=int, default=5, help="Epochs before reducing lr")
parser.add_option("--stop_epoch", type=int, default=10, help="Epochs before reducing lr")
parser.add_option("--lr", type=float, default=3e-4, help="learning rate")

#Model parameters
parser.add_option("--num_layers", type=int, default=6, help="Number of transformer layers")
parser.add_option("--num_heads", type=int, default=2, help="Number of transformer heads")

(flags, args) = parser.parse_args()
scale_lr = flags.lr*np.sqrt(hvd.size())

sim_1 = utils.DataLoader(os.path.join(flags.folder, 'genie_10a_pfn_numucc.npz'),flags.batch,hvd.rank(),hvd.size())
sim_2 = utils.DataLoader(os.path.join(flags.folder, 'genie_10b_pfn_numucc.npz'),flags.batch,hvd.rank(),hvd.size())
#Combine both simulation in a single dataset
sim_1.combine([sim_2])


model = Classifier(num_feat=sim_1.num_feat,
                   num_global=sim_1.num_global,
                   num_classes=sim_1.num_classes,
                   num_layers = flags.num_layers,
                   num_heads = flags.num_heads,
                   class_activation= "sigmoid",
                   )
    
opt = keras.optimizers.Lion(
    learning_rate = scale_lr,
    weight_decay=1e-4,
    clipnorm = 1.0
)
opt = hvd.DistributedOptimizer(opt)
model.compile(weighted_metrics=[],
              loss = keras.losses.BinaryCrossentropy(),
              optimizer=opt,
              #run_eagerly=True,
              metrics=['accuracy'],
              experimental_run_tf_function=False,
              )


callbacks=[
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    EarlyStopping(patience=flags.stop_epoch,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',patience=flags.reduce_epoch, min_lr=1e-6),
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scale_lr,
                                             warmup_epochs=flags.warm_epoch, verbose=1,
                                             steps_per_epoch = sim_1.steps_per_epoch),
]

if hvd.rank()==0:
    if not os.path.exists(os.path.join(flags.folder,'checkpoints')):
        os.makedirs(os.path.join(flags.folder,'checkpoints'))
    checkpoint = ModelCheckpoint(
        os.path.join(flags.folder,'checkpoints','classifier.weights.h5'),
        save_best_only=True,mode='auto',
        save_weights_only=True,
        period=1)
    callbacks.append(checkpoint)


X,y = sim_1.make_tfdata()
hist =  model.fit(X,y,
                  epochs=flags.epoch,
                  validation_split=0.2,
                  batch_size=flags.batch,
                  callbacks=callbacks,
                  shuffle=True,
                  verbose=hvd.rank() == 0,
)
                            
