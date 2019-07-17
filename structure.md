Training dataset:
```
- dataset
  - songname1
    - mixture.wav
    - vocals.wav
  - songname2
    - mixture.wav
    - vocals.wav
    ...

- dataset-validation
  - songname1
    - mixture.wav
    - vocals.wav
  - songname2
    - mixture.wav
    - vocals.wav
    ...
```

Separation:
You can specify any song with the `--file` option

Evaluation:
```
- evaluation
  - songname1
    - vocals.wav
    - accompaniment.wav
    - estimated-vocals.wav
    - estimated-accompaniment.wav
  - songname2
    - vocals.wav
    - accompaniment.wav
    - estimated-vocals.wav
    - estimated-accompaniment.wav
    ...
```
