import numpy as np
from tqdm import tqdm as tqdm
from sklearn_transformers import AddTime, LeadLag, pathwiseExpectedSignatureTransform, SignatureTransform
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline


parameters = {'lin_reg__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
              'lin_reg__fit_intercept': [False, True],
              'lin_reg__normalize': [True, False]}
pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])
MSE_test = np.zeros(NUM_TRIALS)

for i in tqdm(range(NUM_TRIALS)):
    pwES = pathwiseExpectedSignatureTransform(order=2).fit_transform(X)
    SpwES = SignatureTransform(order=3).fit_transform(pwES)
    X_train, X_test, y_train, y_test = train_test_split(np.array(SpwES), np.array(y), test_size=0.2,random_state=i)
    model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv, error_score=np.nan)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MSE_test[i] = mean_squared_error(y_pred, y_test)
