from get_pretrained import model_dict, get_model, save_model_weights


def download_all_models():
    for model_name, _ in model_dict.items():
        model = get_model(model_name, 0)

        if model:
            save_model_weights(model, model_name+'_'+str(0))


if __name__ == '__main__':
    download_all_models()