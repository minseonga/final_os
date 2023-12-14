# final_ossw | brain_tumor 
-student id : 20234424 , Name: kang minseong

# 1.import Load Additional Packages
I used KNeighborsCalssifier, SVC, ExtraTreeClassifier, MLPClassifier, VotingClassifier
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
```
# 2.Load Data Points
```python
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('C:\\Users\\min seong\\Downloads\\final_project-2\\tumor_dataset\\Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)


j = 0
for i in range(len(y)):
    if y[i] in labels[j]:
        plt.imshow(images[i])
        plt.title("[Index:{}] Label:{}".format(i, y[i]))
        plt.show()
        j += 1
    if j >= len(labels):
        break
```

# 3.Split data
 ```python
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=0.01, random_state=42)
```

# 4. Classification with Scikit Learn Library
```python
Knn_Clf1 = KNeighborsClassifier(n_neighbors=1)
Knn_Clf2 = KNeighborsClassifier(n_neighbors=1)
Svc_Clf1 = SVC(kernel='rbf', C=12, degree=10,gamma=1.0,probability=True, coef0=6)
Svc_Clf2 = SVC(kernel='rbf', C=10, degree=9,gamma=1.1,probability=True, coef0=5)
Svc_Clf3 = SVC(kernel='rbf', C=10, degree=9,gamma=1.1,probability=True, coef0=5)
Svc_Clf4 = SVC(kernel='rbf', C=10, degree=10,gamma=1.0,probability=True, coef0=5)

Ex_Tree_Clf1 = ExtraTreesClassifier(n_estimators=3, random_state=869,criterion='entropy')
Ex_Tree_Clf2 = ExtraTreesClassifier(n_estimators=3, random_state=9285,criterion='entropy')
Ex_Tree_Clf3 = ExtraTreesClassifier(n_estimators=2000, random_state=8000,max_depth = 30,
                          max_features = 1,
                          min_samples_split = 5,
                          n_jobs = -1,criterion='entropy')

Mlp_Clf1 = MLPClassifier(max_iter=1000,solver='adam',batch_size = 64, alpha=0.4, random_state=0, hidden_layer_sizes=[10,5,10,2])
Mlp_Clf2 = MLPClassifier(max_iter=1000,solver='adam',batch_size = 64, alpha=0.4, random_state=0, hidden_layer_sizes=[10,5,10,2])

vote = VotingClassifier(estimators=[('Knn1',Knn_Clf1),
                                    ('Knn2',Knn_Clf2),
                                    ('Svc1',Svc_Clf1),
                                    ('Svc2',Svc_Clf2),
                                    ('Svc3',Svc_Clf3),
                                    ('Svc4',Svc_Clf4),
                                    ('EXTree1',Ex_Tree_Clf1),
                                    ('EXTree2',Ex_Tree_Clf2), 
                                    ('EXTree3',Ex_Tree_Clf3),
                                    ('Mlp1', Mlp_Clf1), 
                                    ('Mlp2', Mlp_Clf2)])

vote.fit(X_train, y_train)
y_pred = vote.predict(X_test)
```
# 5.evaluating model 
Please test accuracy with ".pickle" file in Elcass.
```python
import pickle
with open('./final_model_20234424.pickle',"rb") as fr:
    model = pickle.load(fr)
y_pred = model.predict(X_test)
```
