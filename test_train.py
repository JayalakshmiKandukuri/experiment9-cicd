from train import train_model

def test_model_training():
    model, accuracy = train_model()

    assert model is not None
    assert 0 <= accuracy <= 1

def test_model_prediction():
    model, accuracy = train_model()

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert len(prediction) == 1
