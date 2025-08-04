def test_dummy_training():
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X = np.random.rand(10, 5)
    y = [0, 1] * 5
    model = RandomForestClassifier()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 10
