import itertools
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from utils.data_preparation import naive_bayes_preprocessing, bert_preprocessing,gpt2_preprocessing,roberta_preprocessing,electra_preprocessing
from models.naive_bayes import NaiveBayes
from models.bert import Bert
from models.gpt2_roberta_electra import GPT2Emotion,Roberta,Electra
import optuna
from transformers import ElectraForSequenceClassification, ElectraTokenizer, get_linear_schedule_with_warmup
# Configuration du logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def nb_grid_search(params):
    results_list = []
    param_combinations = itertools.product(params['alpha_prior'], params['alpha_likelihood'], params['use_bigrams'], params['remove_stopwords'])

    for alpha_prior, alpha_likelihood, use_bigrams, remove_stopwords in param_combinations:
        processed_data, _ = naive_bayes_preprocessing(
            remove_stopwords=remove_stopwords, 
            use_bigrams=use_bigrams
        )
        X_train, y_train = processed_data['train']
        X_val, y_val = processed_data['validation']
        
        nb = NaiveBayes()
        nb.fit(X_train, y_train, alpha_prior=alpha_prior, alpha_likelihood=alpha_likelihood)
        predictions = nb.predict(X_val)
        acc = nb.evaluate_acc(y_val, predictions)
        
        results_list.append({
            'alpha_prior': alpha_prior,
            'alpha_likelihood': alpha_likelihood,
            'use_bigrams': use_bigrams,
            'remove_stopwords': remove_stopwords,
            'val_accuracy': acc
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    results_df.to_pickle('out/nb_grid_search_results.pkl')
    return results_df


def bert_grid_search(params, exp_name):

    ## Data Preparation ##
    bert_processed_data, tokenizer = bert_preprocessing()
    input_ids_train, attention_mask_train, y_train = bert_processed_data['train']
    input_ids_val, attention_mask_val, y_val = bert_processed_data['validation']
    input_ids_test, attention_mask_test, y_test = bert_processed_data['test']

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)
    test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test)

    results_list = []
    param_combinations = itertools.product(params['epochs'], params['batch_size'], params['lr'], params['weight_decay'],
                                           params['fine_tune_last_layers'])

    for epochs, batch_size, lr, weight_decay, fine_tune_last_layers in param_combinations:

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_eval = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        bert = Bert()
        bert.fit(train_loader, epochs, lr, weight_decay=weight_decay, fine_tune_last_layers=fine_tune_last_layers)
        y_pred = bert.predict(val_loader)
        val_accuracy = accuracy_score(y_val, y_pred)

        results_list.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'fine_tune_last_layers': fine_tune_last_layers,
            'val_accuracy': val_accuracy
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    results_df.to_pickle(f'out/_grid_search_results_{exp_name}.pkl')
    return results_df


def objectivegpt2(trial):
    """Objective function for Optuna optimization."""
    # Hyperparameters à optimiser
    lr = trial.suggest_categorical('lr', [5e-5]) #1e-5, 2e-5, 3e-5,
    batch_size = trial.suggest_categorical('batch_size', [32])#8, 16,
    epochs = trial.suggest_categorical('epochs', [1])#3, 5
    weight_decay = trial.suggest_categorical('weight_decay', [0.001])#0.01, 
    fine_tune_last_layers = trial.suggest_categorical('fine_tune_last_layers', [True])

    # Préparation des données
    gpt2_processed_data, _ = gpt2_preprocessing()  # À implémenter
    input_ids_train, attention_mask_train, y_train = gpt2_processed_data['train']
    input_ids_val, attention_mask_val, y_val = gpt2_processed_data['validation']

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Entraînement du modèle
    gpt2_emotion = GPT2Emotion(seed=42)  # Assurez-vous que GPT2Emotion est défini
    gpt2_emotion.fit(train_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, fine_tune_last_layers=fine_tune_last_layers)

    # Évaluation du modèle
    y_pred = gpt2_emotion.predict(val_loader)
    val_accuracy = accuracy_score(y_val, y_pred)

    return val_accuracy

def gpt2_optuna_search(exp_name, n_trials=1):
    """Perform hyperparameter optimization using Optuna."""
    study = optuna.create_study(direction='maximize')
    study.optimize(objectivegpt2, n_trials=n_trials)

    # Sauvegarder les résultats # test
    results_df = study.trials_dataframe()
    results_df.to_pickle(f'out/gpt2_optuna_results_{exp_name}.pkl')

    # Afficher les meilleurs hyperparamètres
    logging.info(f"Best trial: {study.best_trial.params}")
    logging.info(f"Best validation accuracy: {study.best_trial.value}")

    return study.best_trial.params


def roberta_grid_search(params, exp_name):
    ## Data Preparation ##
    roberta_processed_data, tokenizer = roberta_preprocessing()
    input_ids_train, attention_mask_train, y_train = roberta_processed_data['train']
    input_ids_val, attention_mask_val, y_val = roberta_processed_data['validation']
    input_ids_test, attention_mask_test, y_test = roberta_processed_data['test']

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)
    test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test)

    results_list = []
    param_combinations = itertools.product(params['epochs'], params['batch_size'], params['lr'], params['weight_decay'],
                                           params['fine_tune_last_layers'])

    for epochs, batch_size, lr, weight_decay, fine_tune_last_layers in param_combinations:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        roberta = Roberta()
        roberta.fit(train_loader, epochs, lr, weight_decay=weight_decay, fine_tune_last_layers=fine_tune_last_layers)
        y_pred = roberta.predict(val_loader)
        val_accuracy = accuracy_score(y_val, y_pred)

        results_list.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'fine_tune_last_layers': fine_tune_last_layers,
            'val_accuracy': val_accuracy
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    results_df.to_pickle(f'out/roberta_grid_search_results_{exp_name}.pkl')
    return results_df


def objective(trial):
    """Objective function for Optuna optimization."""
    # Hyperparameters à optimiser
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs = trial.suggest_int('epochs', 3, 5)
    weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.1, log=True)
    fine_tune_last_layers = trial.suggest_categorical('fine_tune_last_layers', [True, False])

    # Préparation des données
    electra_processed_data, _ = electra_preprocessing()
    input_ids_train, attention_mask_train, y_train = electra_processed_data['train']
    input_ids_val, attention_mask_val, y_val = electra_processed_data['validation']

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Entraînement du modèle
    electra = Electra(seed=42)
    electra.fit(train_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, fine_tune_last_layers=fine_tune_last_layers)

    # Évaluation du modèle
    y_pred = electra.predict(val_loader)
    val_accuracy = accuracy_score(y_val, y_pred)

    return val_accuracy

def electra_optuna_search(exp_name, n_trials=20):
    """Perform hyperparameter optimization using Optuna."""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Sauvegarder les résultats
    results_df = study.trials_dataframe()
    results_df.to_pickle(f'out/electra_optuna_results_{exp_name}.pkl')

    # Afficher les meilleurs hyperparamètres
    logging.info(f"Best trial: {study.best_trial.params}")
    logging.info(f"Best validation accuracy: {study.best_trial.value}")

    return study.best_trial.params


if __name__ == '__main__':

    nb_search = False
    bert_search = False

    # Naive Bayes Grid Search
    if nb_search:
        params = {
            'alpha_prior': [0, 1, 10, 100, 1000, 10000, 100000],
            'alpha_likelihood': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            'use_bigrams': [True, False],
            'remove_stopwords': [True, False]
        }
        num_combinations = len(list(itertools.product(*params.values())))
        print(f"The number of combinations is: {num_combinations}")
        grid_search_results = nb_grid_search(params)

    # BERT grid search
    if bert_search:
        params = {
            'epochs': [3],
            'batch_size': [32, 64, 128],
            'lr': [1e-5, 2e-5, 3e-5],
            'weight_decay': [0.01, 0.001],
            'fine_tune_last_layers': [True]
        }
        num_combinations = len(list(itertools.product(*params.values())))
        print(f"The number of combinations is: {num_combinations}")
        grid_search_results = bert_grid_search(params, exp_name='last_layers')
    # GPT2 grid search
    gpt2_emotion_search = False

    logging.info("Starting ELECTRA search")
        
    if gpt2_emotion_search:
        #params = {
         #   'epochs': [1],#, 5
          #  'batch_size': [8],#, 16, 32
           # 'lr': [1e-5],#, 2e-5, 3e-5, 5e-5
           # 'weight_decay': [0.01],#, 0.001
           # 'fine_tune_last_layers': [True]#, False
        #}
        #num_combinations = len(list(itertools.product(*params.values())))
        #print(f"The number of combinations is: {num_combinations}")
        logging.info("GPT2 search enabled")
        # Utilisation de Optuna pour l'optimisation
        best_params = gpt2_optuna_search(exp_name='emotion_generation', n_trials=20)
        print("Best hyperparameters:", best_params)
        logging.info(f"Best hyperparameters: {best_params}")
        
    

    # Roberta grid search
    roberta_search = True

    if roberta_search:
        params = {
            'epochs': [3,5,10],#
            'batch_size': [32,64],#32,
            'lr': [3e-5, 2e-5,5e-5],# 
            'weight_decay': [0.01,0.001,0.003,0.0001],#
            'fine_tune_last_layers': [True, False]
        }
        num_combinations = len(list(itertools.product(*params.values())))
        print(f"The number of combinations is: {num_combinations}")
        grid_search_results = roberta_grid_search(params, exp_name='emotion_classification')
        print(grid_search_results.head())

    # Electra grid search
    electra_search = False
    logging.info("Starting ELECTRA search")

    if electra_search:
        logging.info("ELECTRA search enabled")
        best_params = electra_optuna_search(exp_name='emotion_classification', n_trials=20)
        logging.info(f"Best hyperparameters: {best_params}")
