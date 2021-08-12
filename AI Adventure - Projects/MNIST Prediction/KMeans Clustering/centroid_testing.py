cluster_tests = [10,16,36,64,144,256]
for i in cluster_tests:
  total_clusters = len(np.unique(y_test))
  kmeans = MiniBatchKMeans(n_clusters = i)
  
  kmeans.fit(X_train)
  calculate_metrics(kmeans,y_train)

  reference_labels = retrieve_info(kmeans.labels_, y_train)
  number_labels = np.random.rand(len(kmeans.labels_))
  for i in range(len(kmeans.labels_)):
    number_labels[i] = reference_labels[kmeans.labels_[i]]
  
  print(f'Accuracy: {accuracy_score(number_labels,y_train)}')
  print('\n')


# show guesses compared to test
centroids = kmeans.cluster_centers_
# centroids.shape
centroids = centroids.reshape(256, 28, 28)
centroids *= 255

plt.figure(figsize=(20,18))
bottom = 0.35

for i in range(64):
 plt.subplots_adjust(bottom)
 plt.subplot(8,8,i+1)
 plt.title('Number:{}'.format(reference_labels[i]),fontsize = 17)
 plt.imshow(centroids[i])
