from t_tf.tf_tool import *

layers = tf.keras.layers
losses = tf.keras.losses

layerEmbeddingCenter = layers.Embedding(input_dim=20, output_dim=10)
layerEmbeddingContext = layers.Embedding(input_dim=20, output_dim=10)

# 输入数据, 是分batch的, 这个是batch_size=3
batch_size = 3
centers = [1, 2, 3]
contexts = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
labels = [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]

centers, contexts, labels = convert_tensor(datas=[centers, contexts, labels], types=[tf.int32, tf.int32, tf.float32])

centers_v = layerEmbeddingCenter(centers)
contexts_v = layerEmbeddingContext(contexts)

init_logits = None
for i in range(batch_size):
    sample_logits = tf.matmul(tf.reshape(centers_v[i], (1, -1)), tf.transpose(contexts_v[i]))
    sample_logits = tf.reshape(sample_logits, (-1,))
    if init_logits == None:
        init_logits = sample_logits
    else:
        init_logits = tf.concat([init_logits, sample_logits], axis=0)
labels = tf.reshape(labels, (-1,))

bin_loss = losses.binary_crossentropy(y_true=labels, y_pred=init_logits, from_logits=True)
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(bin_loss)

with tf.Session() as session:
    init_val = tf.compat.v1.global_variables_initializer()
    init_table = tf.compat.v1.tables_initializer()
    session.run([init_table, init_val])

    step = 1000
    for i in range(step):
        _, loss_, center_embed = session.run([train_op, bin_loss, layerEmbeddingCenter.trainable_weights])
        if i % 100 == 0:
            print(f"[step {i}, loss={loss_}, embed-shape={center_embed[0].shape},  embed={sum(list(center_embed[0].reshape(-1)))}]")



# look_var_arr = [
#     init_logits,
#     bin_loss,
#     train_op
#     # centers_v,
#     # contexts_v
# ]
#
#
#
# look_var(look_var_arr)
