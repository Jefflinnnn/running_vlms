from PIL import Image
import io
import pandas as pd 

# Download from hugging face cli
# 1) To do so, follow these steps: https://huggingface.co/docs/huggingface_hub/en/guides/cli
# 2) You may want to add your hf token as an environment variable
# powershell --> [System.Environment]::SetEnvironmentVariable("HF_TOKEN", "your_secret_value_here", "Us")
# mac --> nano ~/.zshrc --> export HF_token="your_secret_value_here
def hf_login():
    import os
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

def hf_dataloader(dataset_name="default_test", see_all  = False):
    import pandas as pd
    
    if see_all:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
    
    # If error, may need to create HF_TOKEN
    # Login using e.g. `huggingface-cli login` to access this dataset
    if dataset_name == "default_test":
        df = pd.read_parquet("hf://datasets/X-iZhang/CheXpert-plus-RRG/findings_section/valid-00000-of-00001.parquet")
    
    return df 
    
if __name__ == "__main__":
    hf_login()
    df = hf_dataloader(see_all = True)

    single_image = df['main_image'].iloc[0]
    single_image_pil = Image.open(io.BytesIO(single_image['bytes']))
    single_image_pil.show()  # This will open the image in the default image viewer
    print(single_image_pil)