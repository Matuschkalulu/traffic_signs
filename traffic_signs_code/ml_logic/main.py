from traffic_signs_code.params import *
from traffic_signs_code.ml_logic.data import create_dataset_with_split, visualize_data
from traffic_signs_code.ml_logic.preprocessing import train_test_preproc, data_augment
from traffic_signs_code.ml_logic.model import resnet_model, train_augment, plot_history, evaluate_model, report
from traffic_signs_code.ml_logic.miscfunc import load_model, save_model

if __name__ == '__main__':
    model_path= os.path.join(LOCAL_MODEL_PATH, 'train_model')
    train_data= create_dataset_with_split(SPLIT_TRAIN_PATH)
    test_data= create_dataset_with_split(SPLIT_TEST_PATH)
    X_train_preproc, X_val_preproc, X_test_preproc, y_train, y_val, y_test= train_test_preproc(train_data, test_data)
    augment_model= resnet_model(X_train_preproc)
    train_flow= data_augment(X_train_preproc, y_train, batch_size=16)
    model, history = train_augment(augment_model, train_flow, X_val_preproc, y_val, epochs=1, batch_size=16, patience=20)
    save_model(model, model_path)
    plot_history(history)
    model= load_model(model_path)
    score= evaluate_model(model, X_test_preproc, y_test)
    report(model, X_test_preproc, y_test)
