from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers

def Classifier(
        num_feat,
        num_global,
        num_classes=1,
        num_layers = 6,
        num_heads=2,
        projection_dim=128,
        class_activation = None,
):


    particles = layers.Input((None,num_feat),name='input_features')
    masked_features = layers.Masking(mask_value=0.0,name='Mask')(particles)
    masked_features = layers.Dense(2*projection_dim,activation="gelu")(masked_features)
    masked_features = layers.Dense(projection_dim,activation="gelu")(masked_features)

    #Lets map the global features using MLPs and combine with particles
    global_event = layers.Input((num_global),name='input_global')

    global_encoded = layers.Dense(2*projection_dim,activation="gelu")(global_event)
    global_encoded = layers.Dense(projection_dim,activation="gelu")(global_encoded)

    #Reshape global inputs to be able to combine with the particles
    global_encoded = layers.Reshape((1,-1))(global_encoded)
    global_encoded = tf.tile(global_encoded,(1,tf.shape(particles)[1],1))

    
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    masked_features = layers.Add()([masked_features,global_encoded])
        
    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(masked_features)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, masked_features])
            
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)

        # Skip connection 2.
        masked_features = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(masked_features)
    #Take the average of all particles before the output
    representation_mean = layers.GlobalAvgPool1D()(representation)
    
    representation_mean =  layers.Dense(2*projection_dim,activation="gelu")(representation_mean)
    outputs = layers.Dense(num_classes,activation=class_activation,)(representation_mean)

    model = Model(inputs=[particles,global_event],
                                outputs=outputs)
    
    return  model


