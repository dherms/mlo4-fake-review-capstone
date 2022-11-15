import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Dataset,
)
from kubernetes.client.models import V1EnvVarSource, V1EnvVar, V1SecretKeySelector

@component(
    packages_to_install=['pandas', 'dvc==2.10.2', 'dvc[s3]'],
    output_component_file='dvc_downloader.yaml'
)
def dvc_downloader(output_csv: Output[Dataset], repo_url :str, rev:str = "main"):
    import pandas as pd
    import dvc.api


    with dvc.api.open(repo=repo_url, path='data/fake reviews dataset.csv', rev=rev) as f:
        df = pd.read_csv(f)
    
    df.to_csv(output_csv.path, header=False, index=False)

@component(
    packages_to_install=['pandas'],
    output_component_file='punctuation_to_features.yaml'
)
def punctuation_to_features(data: Input[Dataset], output_csv: Output[Dataset], column:str ='text'):
    import pandas as pd

    df = pd.read_csv(data.path)
    df[column] = df[column].replace('!', ' exclamation ')
    df[column] = df[column].replace('?', ' question ')
    df[column] = df[column].replace('\'', ' quotation ')
    df[column] = df[column].replace('\"', ' quotation ')

    df.to_csv(output_csv.path, index=False, header=False)

@component(
    packages_to_install=['pandas', 'nltk'],
    output_component_file='tokenize.yaml'
)
def tokenize(data: Input[Dataset], output_csv: Output[Dataset], column:str ='text'):
    import nltk
    import pandas as pd

    nltk.download('punkt')

    df = pd.read_csv(data.path)

    def tokenize_row(row):
        tokens = nltk.word_tokenize(row)
        return [w for w in tokens if w.isalpha()]

    df['tokenized'] = df.apply(lambda x: tokenize_row(x[column]), axis=1)

    df.to_csv(output_csv.path, index=False, header=False)

@component(
    packages_to_install=['pandas', 'nltk'],
    output_component_file='remove_stopwords.yaml'
)
def remove_stopwords(data: Input[Dataset], output_csv: Output[Dataset], tokenized_column:str ='tokenized'):
    import nltk
    from nltk.corpus import stopwords
    import pandas as pd

    df = pd.read_csv(data.path)

    nltk.download('stopwords')

    def remove_row_stopwords(row):
        stops = set(stopwords.words("english"))
        return [word for word in row if not word in stops]

    df['stopwords_removed'] = df.apply(lambda x: remove_row_stopwords(x[tokenized_column]), axis=1)

    df.to_csv(output_csv.path, index=False, header=False)

@component(
    packages_to_install=['pandas', 'nltk'],
    output_component_file='apply_stemming.yaml'
)
def apply_stemming(data: Input[Dataset], output_csv: Output[Dataset], tokenized_column:str ='stopwords_removed'):
    import nltk
    from nltk.stem.porter import PorterStemmer
    import pandas as pd

    df = pd.read_csv(data.path)

    def apply_stemming_row(row):
        stemmer = PorterStemmer()
        return [stemmer.stem(word).lower() for word in row]

    df['porter_stemmed'] = df.apply(lambda x: apply_stemming_row(x[tokenized_column]), axis=1)

    df.to_csv(output_csv.path, index=False, header=False)

@component(
    packages_to_install=['pandas'],
    output_component_file='rejoin_words.yaml',
)
def rejoin_words(data: Input[Dataset], output_csv: Output[Dataset], tokenized_column:str ='porter_stemmed'):
    import pandas as pd

    df = pd.read_csv(data.path)

    def rejoin_words_row(row):
        return  " ".join(row)

    df['all_text'] = df.apply(lambda x: rejoin_words_row(x[tokenized_column]), axis=1)

    df.to_csv(output_csv.path, index=False, header=False)

@dsl.pipeline(
    name='fake-review-preprocessing',
    # You can optionally specify your own pipeline_root
    # pipeline_root='gs://my-pipeline-root/example-pipeline',
)
def my_pipeline(repo_url: str):
    aws_access_key_id_env_var = V1EnvVar("AWS_ACCESS_KEY_ID", value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name='aws-creds', key='aws-access-key-id')))
    aws_secret_access_key_env_var = V1EnvVar("AWS_SECRET_ACCESS_KEY", value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name='aws-creds', key='aws-secret-access-key'))) 
    dvc_downloader_task = dvc_downloader(repo_url)\
        .add_env_variable(aws_access_key_id_env_var)\
        .add_env_variable(aws_secret_access_key_env_var)
    punctuation_to_features_task=punctuation_to_features(data=dvc_downloader_task.outputs['output_csv'])
    tokenize_task=tokenize(data=punctuation_to_features_task.outputs['output_csv'])
    remove_stopwords_task=remove_stopwords(data=tokenize_task.outputs['output_csv'])
    apply_stemming_task=apply_stemming(data=remove_stopwords_task.outputs['output_csv'])
    rejoin_words_task=rejoin_words(data=apply_stemming_task.outputs['output_csv'])

kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func=my_pipeline,
    package_path='pipeline.yaml')