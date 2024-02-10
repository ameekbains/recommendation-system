import numpy as np
import tensorflow as tf

# Sample user-item matrix
user_item_matrix = np.array([
    [1, 0, 3, 0, 2],
    [4, 0, 0, 0, 5],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [2, 0, 0, 0, 4]
])

# Number of users and items
num_users, num_items = user_item_matrix.shape

# Convert the user-item matrix to a tensor
user_item_tensor = tf.constant(user_item_matrix, dtype=tf.float32)

# Latent factors dimensionality
latent_dim = 2

# User embeddings and item embeddings
user_embeddings = tf.Variable(tf.random.normal([num_users, latent_dim]))
item_embeddings = tf.Variable(tf.random.normal([num_items, latent_dim]))

# Compute predictions
predictions = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)

# Masking for known ratings
masked_predictions = tf.where(tf.equal(user_item_tensor, 0),
                              tf.zeros_like(predictions),
                              predictions)

# Loss function
loss = tf.reduce_sum(tf.square(masked_predictions - user_item_tensor))

# Optimization
optimizer = tf.optimizers.Adam(learning_rate=0.01)
train_step = optimizer.minimize(loss)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    train_step.run()
    if epoch % 100 == 0:
        print("Epoch {}, Loss: {}".format(epoch, loss.numpy()))

# Get the learned embeddings
learned_user_embeddings = user_embeddings.numpy()
learned_item_embeddings = item_embeddings.numpy()

# Calculate recommendations for a user (e.g., user 0)
user_index = 0
user_ratings = user_item_matrix[user_index]
user_embedding = learned_user_embeddings[user_index]
recommendations = np.dot(learned_item_embeddings, user_embedding)
sorted_recommendations_indices = np.argsort(recommendations)[::-1]

print("Recommendations for User {}: ".format(user_index))
for i in sorted_recommendations_indices:
    if user_ratings[i] == 0:
        print("Item {}: {}".format(i, recommendations[i]))
