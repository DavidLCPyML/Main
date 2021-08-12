# make model
num_clusters = len(np.unique(y_test))
kmeans = MiniBatchKMeans(n_clusters = num_clusters)
kmeans.fit(X_train)

# output labels created
print(      kmeans.labels_[0:]      )
print(      kmeans.labels_    .shape)
print(type( kmeans.labels_          ))
print(      kmeans.labels_          )


# extract labels from information
def retrieve_info(cluster_labels,y_train):
  # Initializing
  reference_labels = {}
  # For loop to run through each label of cluster label
  for i in range(len(np.unique(kmeans.labels_))):
    index = np.where(cluster_labels == i,1,0)
    num = np.bincount(y_train[index==1]).argmax()
    reference_labels[i] = num
  return reference_labels

def calculate_metrics(model, output): 
  # text = {model.n_clusters, model.inertia_, metrics.homogeneity_score(output, model.labels)}
  print(f'Clusters: {model.n_clusters}')
  print(f'Inertia: {model.inertia_}')
  print(f'Homogeneity: {metrics.homogeneity_score(output, model.labels_)}')



# extract clusters from labels
reference_labels = retrieve_info(kmeans.labels_,y_train)
print(reference_labels)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
  number_labels[i] = reference_labels[kmeans.labels_[i]]

# Comparing Predicted values and Actual values
print(number_labels[:20].astype('int'))
print(y_train[:20])
# Calculating accuracy score
print(accuracy_score(number_labels,y_train))
