# Ponderada-Semana02-25 — README explicativo do notebook (Transformer PT→EN)

## 1) Objetivo da atividade
Entregar um **README, em português**, que **explique o notebook** `notebooks/transforme.ipynb`, destacando:
- O que cada parte do notebook faz,
- Como executar,
- Dicas práticas e armadilhas comuns.

---


## 2) Como executar o notebook
- **Google Colab (recomendado):** abra `notebooks/transforme.ipynb` e rode as células na ordem.
- **Ambiente local (opcional):**
  ```bash
  pip install "tensorflow==2.16.*" "tensorflow-text==2.16.*" tensorflow-datasets matplotlib numpy

---

## 3) O que o notebook faz (passo a passo)

### 3.1 Setup
- Importa bibliotecas: `tensorflow`, `tensorflow_text`, `tensorflow_datasets`, `numpy`, `matplotlib`.
- (Opcional) cria diretórios auxiliares (ex.: `results/`) para salvar gráficos/tempos.

**Por que importa?** Garante ambiente consistente e pronto para o restante.

---

### 3.2 Dados (TFDS: `ted_hrlr_translate/pt_to_en`)
- Carrega pares PT→EN diretamente do **TensorFlow Datasets**:
  ```python
  import tensorflow_datasets as tfds

  train_examples, val_examples = tfds.load(
      "ted_hrlr_translate/pt_to_en",
      split=["train","validation"],
      as_supervised=True
  )
- Para prototipagem rápida (principalmente em CPU), é comum reduzir:
    ```python
    train_examples = train_examples.take(2000)
    val_examples   = val_examples.take(200)

**Por que importa?** TFDS padroniza acesso aos dados e facilita reprodutibilidade.

---

### 3.3 Tokenização (SavedModel oficial)

O notebook usa **tokenizers prontos** (PT e EN) publicados como **SavedModel**. Fluxo típico:
1. Baixar o arquivo **`.zip`** dos tokenizers;  
2. **Extrair**;  
3. Localizar a pasta que contém **`saved_model.pb`**;  
4. **Carregar** com `tf.saved_model.load(...)`.

**Python (download + extração + carga)**
    ```python
    import os, glob, tensorflow as tf
    import tensorflow_text as tf_text  # mantenha a mesma versão do TF

    zip_path = tf.keras.utils.get_file(
        fname="ted_hrlr_translate_pt_en_converter.zip",
        origin="https://storage.googleapis.com/download.tensorflow.org/models/ted_hrlr_translate_pt_en_converter.zip",
        extract=True
    )

    root = os.path.dirname(zip_path)              # onde o Keras extraiu
    matches = glob.glob(os.path.join(root, "**", "saved_model.pb"), recursive=True)
    assert matches, f"Nenhum saved_model.pb encontrado em {root}"
    model_dir = os.path.dirname(matches[0])

    tokenizers = tf.saved_model.load(model_dir)

---

### 3.4 Pipeline `tf.data`

- Tokeniza PT/EN, **limita o comprimento** (`MAX_TOKENS`) e faz **shift** do alvo:
  - `en_in = en[:, :-1]` (entrada do decoder)
  - `en_out = en[:, 1:]` (rótulo a prever)
- Constrói dataset eficiente com `cache → shuffle → padded_batch → prefetch`.

    ```python
    import tensorflow as tf

    MAX_TOKENS = 128
    BATCH_SIZE = 32
    BUFFER = 20_000

    def prepare_example(pt, en):
        pt = tokenizers.pt.tokenize(pt)[..., :MAX_TOKENS]
        en = tokenizers.en.tokenize(en)[..., :MAX_TOKENS]
        en_in  = en[:, :-1]
        en_out = en[:,  1:]
        return (pt, en_in), en_out

    def make_dataset(examples):
        return (examples
                .map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .shuffle(BUFFER)
                .padded_batch(
                    BATCH_SIZE,
                    padded_shapes=(([None, None], [None, None]), [None, None])
                )
                .prefetch(tf.data.AUTOTUNE))

    ds_train = make_dataset(train_examples)
    ds_val   = make_dataset(val_examples)
    ```

**Por que importa?** `tf.data` evita gargalos de I/O e `padded_batch` permite treinar com sequências de tamanhos diferentes.

---

### 3.5 Modelo Transformer (Keras)

- **Positional Encoding** + **Embedding**
- **Multi-Head Attention**: self-attention (encoder), **causal** (decoder) e **cross-attention**
- **Feed-Forward** com residual + layer norm
- **Encoder** (pilhas) e **Decoder** (pilhas) com projeção final para o vocabulário
- Para `model.summary()` em modelos *subclassed*, faça antes uma passada **dummy**.

    ```python
    # Passada "dummy" para inicializar pesos e permitir summary()
    _ = transformer(
        (tf.ones((1, 10), tf.int64),  # pt (encoder)
        tf.ones((1,  9), tf.int64)), # en_in (decoder)
        training=False
    )
    # transformer.summary()  # opcional
    ```

**Por que importa?** Em modelos *subclassed*, a passada *dummy* cria os pesos e evita erro no `model.summary()`.

---

### 3.6 Treinamento

- **Otimizador:** `Adam`
- **Perda:** `SparseCategoricalCrossentropy(from_logits=True)` com **máscara** para ignorar *padding*
- Fluxo: `compile(...)` → `fit(ds_train, validation_data=ds_val, epochs=...)`

    ```python
    # Loss com máscara + compile
    base_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def masked_loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)  # 0 = padding
        loss = base_loss(y_true, y_pred) * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    transformer.compile(optimizer=optimizer, loss=masked_loss)
    ```

    ```python
    # Callback simples para medir tempos + fit
    import time
    import tensorflow as tf

    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.t0 = time.perf_counter(); self.epoch_times = []
        def on_epoch_begin(self, epoch, logs=None):
            self.t1 = time.perf_counter()
        def on_epoch_end(self, epoch, logs=None):
            self.epoch_times.append(time.perf_counter() - self.t1)
        def on_train_end(self, logs=None):
            self.total_time = time.perf_counter() - self.t0

    time_cb = TimeHistory()

    history = transformer.fit(
        ds_train,
        validation_data=ds_val,
        epochs=3,              # ajuste conforme CPU/GPU
        callbacks=[time_cb]
    )
    ```

    ```bash
    # instalação local rápida
    pip install "tensorflow==2.16.*" "tensorflow-text==2.16.*" tensorflow-datasets matplotlib numpy
    ```

**Por que importa?** A máscara impede que o padding distorça a perda; o callback registra tempos para comparar CPU × GPU.

---

## 4) O que observar

- **Loss** tende a **cair** ao longo das épocas (mesmo com dataset reduzido).
- Traduções **melhoram** com **mais dados/épocas**.
- Em **GPU** o treino é **muito mais rápido** do que em CPU.

**Por que importa?** São sinais de aprendizado correto e mostram o impacto do hardware no tempo total.

---

## 5) Problemas e soluções rápidas

- **Tokenizers (.zip):** extraia e aponte para a pasta que contém `saved_model.pb`.
- **Erro no `tf.saved_model.load(...)`:** confirme o caminho após a extração (busca recursiva).
- **`transformer.summary()` falha:** faça a passada *dummy* (ver 3.5).
- **`tensorflow-text` incompatível:** use a **mesma versão** do TensorFlow.

    ```bash
    # verificar GPU no ambiente
    nvidia-smi || echo "Sem GPU NVIDIA disponível"
    ```

**Por que importa?** Esses passos resolvem os erros mais comuns e garantem reprodutibilidade.

---

## 6) CPU vs GPU (Não consegui rodar)

### Como reproduzir
- **CPU**: antes de importar o TensorFlow no notebook:
    ```python
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # desativa GPU
    import tensorflow as tf
    ```
- **GPU (Colab)**: ative GPU no ambiente (`Runtime → Change runtime type → GPU`) e rode normalmente.
- (Opcional) use o callback `TimeHistory` (seção 3.6) para medir tempo por época.

**Tabela**

| Item                     | CPU (ex.) | GPU (ex.) |
|--------------------------|-----------|-----------|
| Tempo/época (média)      | ...       | ...       |
| Tempo total              | ...       | ...       |
| Loss final (train/val)   | ... / ... | ... / ... |

*Tendência:* GPU costuma ser várias vezes mais rápida, dependendo do tamanho do modelo/lote.

---

## 7) Percepções pessoais (prós e desafios)

**Prós**
- Keras organiza bem as partes do Transformer.
- `tf.data` simplifica desempenho com cache/shuffle/prefetch.
- Separação Encoder/Decoder/Attention ajuda a entender a arquitetura.

**Desafios**
- Compatibilidade de versões (`tensorflow` / `tensorflow-text`).
- Treino pode ser lento sem GPU.
- Ajustes de hiperparâmetros impactam muito a qualidade.

---

## 8) Próximos passos
- Usar **mixed precision** em GPU para acelerar.
- Treinar mais épocas/dados e medir **BLEU**.
- Testar **KerasNLP** ou tokenizadores próprios (SentencePiece).
- Adicionar **label smoothing** e **warmup schedule** do paper original.

---

## 9) Minha experiência nesta atividade

**Resumo:** Não consegui rodar o notebook de ponta a ponta por limitações técnicas (download do tokenizer e compatibilidade de versões), mas **estudei o tutorial oficial**, revisei o código e documentei o fluxo completo. Abaixo, o que tentei e o que aprendi.

### O que eu tentei
- Baixar e carregar os tokenizers `ted_hrlr_translate_pt_en_converter` (SavedModel) para PT/EN.
- Carregar com `tf.saved_model.load(...)` após a extração.
- Ajustar compatibilidade instalando `tensorflow-text` com a **mesma versão** do `tensorflow`.
- Validar o pipeline com `tf.data` (map → cache → shuffle → padded_batch → prefetch).

### O que aprendi (mesmo sem executar tudo)
- **TFDS + tf.data:** como montar um pipeline eficiente e batelado, e por que o `padded_batch` é necessário em NLP.
- **Tokenização via SavedModel:** a importância de extrair corretamente e **apontar para a pasta que contém `saved_model.pb`** (erros comuns: 403 no download, caminho errado após extração).
- **Arquitetura do Transformer:** encoder/decoder, **self-attention global** (encoder), **self-attention causal** (decoder), **cross-attention**, positional encoding e residual + layer norm.
- **Treinamento:** perda com máscara para ignorar padding (Sparse Categorical CE `from_logits=True`), ideia de **label smoothing** e **scheduler** de learning rate do paper.
- **Build antes do summary:** por ser `Model` subclassed, precisa de uma passada “dummy” para `model.summary()`.
- **CPU vs GPU:** conceitualmente, GPU reduz muito o tempo por época; CPU serve para testar pipeline e hiperparâmetros menores.

### Principais obstáculos (e causas prováveis)
- **403/404 ao baixar os tokenizers:** ambiente bloqueando `storage.googleapis.com` ou link `.tgz` instável.
- **`SavedModel` não encontrado:** pasta errada após a extração (underscores, níveis de diretório, etc.).
- **Incompatibilidade de versões:** `tensorflow-text` diferente do `tensorflow`.
- **Tempo de treino elevado em CPU:** hiperparâmetros grandes e dataset completo.
