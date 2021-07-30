# WeCanBeLGMan

## 시도 했던 내용

segmentation_models_pytorch

encoder(backbone)
- efficientent
- xceptionet

decoder
- Unet
- Unet++
- DeepLabV3
- DeepLabV3+
- PAN

albumentation
- flip (horizontal, vertical)
- transpose
- rotation90

data loader
- train과 validation에는 256 이미지 사이즈와 128 스트라이드를 적용해서 원본 이미지를 sliding window 형태로 잘라서 사용했으며, 데이콘에 제출하는 테스트 이미지의 경우, 256 이미지 사이즈와 32 스트라이드를 적용

optimizer
- adam
- adamw
- sgd (with momentum & nesterov)

optimizer model parameter
- model.parameters()
- "encoder": lr * 0.1, "decoder": lr

encoder(backbone) 모델에 lr의 0.1을 곱한 값을 적용한 것보다 전체 모델 파라미터에 lr를 적용해서 학습 시킨 모델이 성능이 더 높게 나왔음

성능 상으로는 efficientnet-b1과 Unet++ 를 사용했을 때 public psnr_score 28.6을 기록

나머지 테스트의 경우 처음 모델보다 현저하게 떨어진 psnr 성능을 나타냈다.

