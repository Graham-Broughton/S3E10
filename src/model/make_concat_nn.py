# number of nodes
def n_nodes(value):
    return (value - 1)


# generate the table of layers
def gen_nn(name, old_name, nodes, width, depth, f, list_concat):
    if depth == 0:
        f.write(
            '\t' + old_name + '= tf.keras.layers.Dense(' + str(nodes) + ', activation=tf.nn.swish, name = "' + old_name + '")(' + old_name + ')' + '\r'
        )
        f.write('\t' + old_name + ' = layers.BatchNormalization()(' + old_name + ')' + '\r')
        list_concat.append(old_name)
        f.write('\t' + '\r')
        return 0

    f.write('\t' + name + '= tf.keras.layers.Dense(' + str(nodes) + ',activation=tf.nn.swish)(' + old_name + ')' + '\r')
    # f.write('\t' + name + '= layers.BatchNormalization()(' + name + ')'+ '\r')
    # f.write('\t' + name + '= tf.keras.layers.Dropout(0.2)(' + name + ')'+ '\r')
    # do the same for the next depth, there is 'width' outputs
    # only do one output when depth == 0
    for i in range(width):
        str_next = f'{name}_{str(i)}'
        old_next = name
        nodes_next = n_nodes(nodes)
        gen_nn(str_next, old_next, nodes_next, width, depth - 1, f, list_concat)
        nodes = n_nodes(nodes)
        if depth == 1:
            break


# init
list_concat = []
number_start_layers = 20
width = 2
depth = 5
with open("include_concat.py", 'w') as f:
    f.write('def NN(x_train):' + '\r')
    f.write('\t' + 'inputs = layers.Input(shape=(x_train.shape[-1],))' + '\r')
    # f.write('\t' + 'inputs = layers.BatchNormalization()(inputs)'+ '\r')
    f.write('\t' + 'x = tf.keras.layers.Dense(' + str(number_start_layers) + ', activation=tf.nn.swish)(inputs)' + '\r')
    f.write('\t' + 'x = layers.BatchNormalization()(x)' + '\r')
    f.write('\t' + 'x = tf.keras.layers.Dropout(0.2)(x)' + '\r')
    # generate
    gen_nn('layer_0', 'x', number_start_layers, width, depth, f, list_concat)

    # end
    f.write('\t' + 'concat = tf.keras.layers.concatenate([' + ', '.join(list_concat) + '])' + '\r')
    f.write('\t' + 'output0 = tf.keras.layers.Dense(' + str(number_start_layers) + ', activation=tf.nn.swish)(concat)' + '\r')
    f.write('\t' + 'output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output0)' + '\r')
    f.write('\t' + 'ensemble = tf.keras.models.Model(inputs, output)' + '\r')
    f.write('\t' + 'plot_model(ensemble, to_file="model_concat.png", show_shapes=True)' + '\r')
    f.write('\t' + 'return(ensemble)' + '\r')

print("\t\t\t\t\t\tEnd of program\n\t\t\t\t\t\t______________\n\n")
