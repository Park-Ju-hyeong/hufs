# Generative Adversarial Networks

해당 문서는 df형태의 데이터를 생성하는 코드의 사용방법을 다룹니다.

---
## Run

데이터와 코드를 먼저 받아주세요
```
https://github.com/Park-Ju-hyeong/hufs.git
```

```
# Good 데이터 생성
python Agile_VEEGAN.py --num-steps=10000 --batch-size=4000 --label=0 --num-layer=4 --hidden-size=384 --num-gen=10000
```

```
# Bad 데이터 생성
python Agile_VEEGAN.py --num-steps=10000 --batch-size=3000 --label=1 --num-layer=4 --hidden-size=384 --num-gen=10000 
```

### parameter 설명

help를 통해 간단한 설명을 볼 수 있습니다.
```
python Agile_VEEGAN.py --help
```

|   params  |   설명  |
|:--:|:--:|
|--num-steps, type=int, default=5000    |   iteration 회수
|--input-dim, type=int, default=254 |   gen 에 들어가는 정규분포 크기
|--output-dim, type=int, default=44 |   real 데이터의 설명변수 개수
|--eps-dim, type=int, default=1 |   infer 에 들어가는 표준정규분포 크기
|--hidden-size, type=int, default=384   |   gen, inger, disc 히든 노드수
|--num-layer, type=int, default=4  |    gen, inger, disc 히든 레이어수
|--batch-size, type=int, default=3000   |   미니배치 크기
|--log-every, type=int, default=100 |   iteration 몇번마다 log 볼 것인가
|--num-gen, type=int, default=10000 |   학습 후 몇개의 데이터를 생성할 것인가
|--save, type=bool, default=True    |   ouput을 save할 것인가 
|--label, type=int, default=0   |   0: Good 1: Bad 
