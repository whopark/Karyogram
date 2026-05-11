# PRD: Chromosome Karyotype Analyzer - Precision Clinical Lens

## 1. 프로젝트 개요

**프로젝트명**: Chromosome Karyotype Analyzer
**목표**: AI 기반 세포유전학 분석 도구 — 중기 분열상 이미지에서 ISCN 2020 규격의 핵형 표기를 자동 생성
**플랫폼**: Streamlit 웹 애플리케이션 (단일 파일 구조, `app.py` 5,430 라인)
**실행**: `streamlit run karyotype-analyzer-2/app.py`

---

## 2. 로드맵 (roadmap.png) — 정밀 임상 렌즈 (The Precision Clinical Lens)

원시 세포분열 중기 이미지에서 최종 진단 핵형도까지 6단계 순차 파이프라인:

```
원시 이미지 → [1.계수] → [2.분류] → [3.클러스터분류] → [4.전이] → [5.분석] → [6.이상탐지] → 최종 핵형도
```

이 파이프라인은 `PrecisionClinicalLens` 클래스로 구현되어 VLM API를 6회 순차 호출하며, 각 단계가 이전 결과를 기반으로 정밀도를 높여갑니다.

---

## 3. 구현 완료 Task 목록

### Task 1: 염색체 계수 — 객체 탐지 알고리즘의 적용

**문제**: 자기유사성(Self-Similarity)과 극심한 겹침(Severe Overlap)
**해결**:

| 구현 항목 | 방법 | 클래스/메서드 |
|-----------|------|---------------|
| 밴딩 패턴 마이닝 | PCA 주축 따라 32-bin intensity profile 추출 | `ChromosomeDetector._extract_banding_profiles()` |
| 2-Stage NMS | 1단계 공간 IoU → 2단계 밴딩 유사도 검사 (cosine < 0.85이면 보존) | `ChromosomeDetector._two_stage_nms()` |
| 겹침 분리 (Watershed) | Distance transform → 로컬 최대값 → 워터셰드 분할 | `ChromosomeDetector._watershed_split()` |
| 겹침 분리 (Concavity) | 볼록 껍질 결함점 탐지 → 깊은 오목점 절단선 | `ChromosomeDetector._concavity_split()` |
| 다중 임계값 탐지 | Otsu, Adaptive, Multi-level simple threshold 병렬 실행 후 최적 선택 | `ChromosomeDetector._detect_from_threshold()` |

---

### Task 2: 얽힌 매듭 풀기 — 분할 매트릭스 (Segmentation Matrix)

**문제**: 겹친 염색체를 개별 인스턴스로 분리
**해결**: 시멘틱 분할 + 인스턴스 분할 이중 경로

| 경로 | 방법 | 설명 |
|------|------|------|
| **Path 1: 시멘틱 분할** | 픽셀 3-class 분류 (배경/단일/겹침) | 강도 분석으로 겹침 영역 감지, gradient 방향 기반 stitching 복원 |
| **Path 2: 인스턴스 분할** | 마커 제어 워터셰드 | Distance transform 마커 + Sobel gradient 경계 + gradient-enhanced watershed |
| **결합 전략** | 적응형 선택 | 겹침 유의미 → 시멘틱 stitching, 인스턴스가 더 정확 → 인스턴스 채택, 둘 다 아니면 passthrough |

**클래스**: `SegmentationMatrix` (메서드: `semantic_segmentation()`, `instance_segmentation()`, `stitch_from_overlap()`, `segment_and_separate()`)

---

### Task 3: 클러스터 분류 — AI의 사전 라우팅 메커니즘

**문제**: 모든 클러스터에 동일한 분할을 적용하면 연산 낭비 및 정확도 저하
**해결**: 겹침 유형별 사전 분류 → 최적 파이프라인 라우팅

```
입력 클러스터 → [인접성 행렬 구축] → [BFS 연결 컴포넌트] → [4-class 분류] → 라우트 배정
```

| 유형 | 라우트 | 파이프라인 |
|------|--------|-----------|
| Isolated (고립) | Pass-through | 변경 없이 통과 |
| Touching (맞닿음) | Route A | 인스턴스 분할 (마커 watershed) |
| One-Overlap (단일 겹침) | Route B | 시멘틱 stitching (gradient-guided) |
| Multi-Overlap (다중 겹침) | Route C | 다중 패스 (stitching → watershed → concavity) |

**클래스**: `ClusterRouter` (메서드: `classify_clusters()`, `route_and_segment()`, `_route_a_touching()`, `_route_b_one_overlap()`, `_route_c_multi_overlap()`)

---

### Task 4: 디지털 전처리 — 강건화 및 궤적 교정

**문제**: 배경 노이즈, 불순물, 염색체 곡률로 인한 분석 정확도 저하
**해결**: Cascaded Denoising + Medial Axis Straightening

#### Cascaded Denoising (4단계)

| 단계 | 처리 | 조건 |
|------|------|------|
| 1 | Background Subtraction | 조명 편차 > 30일 때만 적용 (적응형) |
| 2 | Debris Removal | 면적 < 0.05% 입자를 inpainting 제거 |
| 3 | CLAHE | 적응형 대비 향상 (clip=3.0, grid=8x8) |
| 4 | Bilateral Filter | 에지 보존 스무딩 (d=7) |

#### Chromosome Straightening

| 단계 | 방법 |
|------|------|
| Skeleton 추출 | Distance transform peak ridge (0.037s/chromosome) |
| 점 정렬 | 최근접 이웃 greedy ordering from endpoint |
| 곡선 평활화 | Moving average + 균일 리샘플링 |
| 직선화 | 주축 수직 방향 intensity strip 샘플링 |
| 곡률 측정 | 이산 곡률 공식 (curvature) |

**클래스**: `DigitalPreprocessor` (메서드: `denoise()`, `straighten_chromosome()`, `straighten_all()`)

---

### Task 5: 24-클래스 분류의 시각적 패러독스

**문제**: 22개 상염색체 + X, Y의 24 클래스 분류 — G-Band 품질 편차, 데이터 불균형
**해결**: 참조 템플릿 기반 특징 매칭 + 초해상도 + Pair-based Refinement

#### 24 참조 템플릿 (ISCN 2020)

각 염색체(1-22, X, Y)에 대해:
- `size_pct`: 전체 게놈 대비 상대 크기 (%)
- `ci`: 동원체 지수 (centromere index)
- `group`: Denver 그룹 (A-G)
- `type`: 형태 (metacentric / submetacentric / acrocentric)

#### 특징 추출 및 분류

| 특징 | 비중 | 방법 |
|------|------|------|
| 크기 비율 | 45% | 전체 면적 대비 비율, Gaussian 유사도 |
| 동원체 지수 | 30% | 최소 폭 탐지 (primary constriction) |
| 밴딩 호환성 | 15% | metacentric/submetacentric/acrocentric 적합도 |
| 종횡비 | 10% | 길이/너비 비율 |

#### Super-Resolution

작은 염색체(Group F, G)를 64px로 bicubic 업스케일 + unsharp masking 후 밴딩 재추출

#### Pair-based Refinement

autosome=2개 제약 조건 적용, 최저 신뢰도 과잉 클래스를 차선 클래스로 재배정 (최대 10회 반복)

**클래스**: `ChromosomeClassifier` (메서드: `classify_all()`, `_extract_features()`, `_estimate_centromere_index()`, `_super_resolve_banding()`, `_compute_class_scores()`, `_refine_with_pairing()`, `generate_karyotype_summary()`)

---

### models.png: 분류 알고리즘의 진화 — 5가지 핵심 방법론 앙상블

**문제**: 단일 분류기의 한계 — 각 방법론이 서로 다른 강점을 가짐
**해결**: 5가지 전략을 가중 다수결 투표로 결합

| # | 전략 | 영감 모델 | 가중치 | 핵심 접근법 |
|---|------|----------|--------|------------|
| 1 | 경량화 CNN | Simple CNN | 15% | 크기 + 동원체만으로 빠른 분류 |
| 2 | 특징 대조 | Siamese Net | 20% | 밴딩 프로필 코사인 거리 + 형태 유형 임베딩 |
| 3 | 전처리 강화 | SRAS-Net | 20% | 초해상도 업스케일 후 특징 재추출 |
| 4 | 전역/국소 융합 | VariFocal-Net | 25% | 전체 형태(solidity, 크기) + 국소 밴딩(dark bands, 대비) |
| 5 | 다중 작업 앙상블 | DeepACC/HRNet | 20% | 분류 × 분할 품질 × 밴딩 품질 × 직선도 |

**투표 메커니즘**: 5개 전략 독립 투표 → 가중 합산 → 최고 점수 클래스 선택 → Pair-based refinement

**클래스**: `EnsembleClassifier` (메서드: `classify_ensemble()`, `_strategy_simple_cnn()`, `_strategy_siamese()`, `_strategy_sras()`, `_strategy_varifocal()`, `_strategy_multitask()`)

---

## 4. 전체 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ChromosomeDetector.detect_chromosomes()          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 0: DigitalPreprocessor.denoise()              [Task 4]      │
│     └─ Background Subtraction → Debris Removal → CLAHE → Bilateral│
│                                                                     │
│  Stage A: Multi-threshold Contour Extraction          [Task 1]      │
│     └─ Otsu + Adaptive + Multi-level + Small-area tolerance        │
│                                                                     │
│  Stage B: ClusterRouter.route_and_segment()           [Task 3]      │
│     ├─ Classify: Isolated / Touching / One-Overlap / Multi-Overlap │
│     ├─ Route A → SegmentationMatrix.instance_segmentation()        │
│     ├─ Route B → SegmentationMatrix.stitch_from_overlap()          │
│     └─ Route C → Multi-pass (stitching + watershed + concavity)    │
│                                                     [Task 2]       │
│  Stage B2: Overlap Split Fallback                     [Task 1]      │
│     └─ Watershed + Concavity split for remaining merges            │
│                                                                     │
│  Stage C: DigitalPreprocessor.straighten_all()        [Task 4]      │
│     └─ Skeleton → Medial axis → Perpendicular sampling → Banding   │
│                                                                     │
│  Stage D: Two-stage NMS                               [Task 1]      │
│     └─ Spatial IoU NMS → Banding-aware NMS                         │
│                                                                     │
│  Stage E: EnsembleClassifier.classify_ensemble()      [Task 5+models]│
│     ├─ Strategy 1: Simple CNN (15%)                                │
│     ├─ Strategy 2: Siamese Contrastive (20%)                       │
│     ├─ Strategy 3: SRAS-Enhanced (20%)                             │
│     ├─ Strategy 4: VariFocal Fusion (25%)                          │
│     ├─ Strategy 5: Multi-Task Ensemble (20%)                       │
│     └─ Weighted Voting + Pair-based Refinement                     │
│                                                                     │
│  Output: count, bounding_boxes, classifications,                    │
│          karyotype_summary (ISCN notation), ensemble metadata       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 클래스 구조 (9개 클래스, 5,430 라인)

| 클래스 | 라인 | 역할 |
|--------|------|------|
| `APIProvider` | 46 | API 제공자 Enum (OpenAI, Anthropic, Gemini, Precision Lens 등) |
| `DigitalPreprocessor` | 57-378 | 노이즈 제거 + 염색체 직선화 |
| `SegmentationMatrix` | 379-767 | 시멘틱 + 인스턴스 분할 이중 경로 |
| `ClusterRouter` | 768-1156 | 겹침 유형별 사전 라우팅 |
| `ChromosomeClassifier` | 1157-1596 | 24-class 특징 기반 분류 + 참조 템플릿 |
| `EnsembleClassifier` | 1597-1993 | 5-strategy 앙상블 투표 |
| `ChromosomeDetector` | 1994-3125 | CV 통합 파이프라인 (Task 1-5 + models) |
| `PrecisionClinicalLens` | 3126-3555 | 6단계 VLM 순차 분석 파이프라인 |
| `KaryotypeAnalyzer` | 3556+ | 다중 API 제공자 라우터 + UI 통합 |

---

## 6. 검증 결과

### CV 파이프라인 (ChromosomeDetector) 테스트

| 테스트 이미지 | 기대값 | 검출 수 | Denver Groups | ISCN |
|--------------|--------|---------|---------------|------|
| Normal NHGRI | 46,XY | **46** | A:4 B:4 C:16 D:6 E:6 F:4 G:6 | 46,X3Y2 |
| Klinefelter | 47,XXY | 44 | A:4 B:4 C:14 D:6 E:6 F:4 G:6 | 44,X2Y2 |
| Down Syndrome | 47,+21 | 52 | - | - |
| Turner | 45,X | 63 | - | - |
| Triple X | 47,XXX | 55 | - | - |

### 24-Class 앙상블 분류 (NHGRI 46,XY)

```
22개 autosome 중 21개 정확히 2개씩 분류
5-strategy unanimous agreement: 28%
Avg confidence: 0.733
```

---

## 7. 분석 모드 (UI)

| 모드 | 설명 |
|------|------|
| OpenAI GPT-4 Vision | VLM 단독 분석 |
| CV + VLM (Hybrid) | CV 카운팅 → VLM 해석 |
| **Precision Clinical Lens (6-Stage)** | 6단계 VLM 순차 파이프라인 (계수→분류→클러스터→전이→분석→이상탐지) |
| Multi-Model Consensus | GPT-4 + Claude + Gemini 다수결 투표 |
| Demo Mode | API 없이 시뮬레이션 |

---

## 8. 기술 스택

| 구분 | 기술 |
|------|------|
| Frontend | Streamlit 1.57+ |
| CV Engine | OpenCV (cv2), NumPy, Pillow |
| VLM API | OpenAI GPT-4o, Anthropic Claude Sonnet 4, Google Gemini 2.0 Flash |
| 표준 | ISCN 2020 (International System for Human Cytogenomic Nomenclature) |

---

## 9. 테스트 이미지 (11개)

```
test_images/
├── nhgri_karyotype.png           # 정상 46,XY (NHGRI 공개)
├── down_syndrome_karyotype.png   # 다운증후군 47,+21
├── klinefelter_47XXY.jpg         # 클라인펠터 47,XXY
├── turner_45X.jpg                # 터너 45,X
├── turner_45X_wellcome.jpg       # 터너 (Wellcome 소스)
├── triple_x_47XXX.jpg            # 트리플X 47,XXX
├── triple_x_47XXX_2.jpg          # 트리플X (변형)
├── triple_x_47XXX_pmc.jpg        # 트리플X (PMC 소스)
├── triple_x_47XXX_cropped.jpg    # 트리플X (크롭)
├── triple_x_47XXX_rg.jpg         # 트리플X (RG)
└── turner_triple_x_combined.jpg  # 터너+트리플X 결합
```

---

## 10. 향후 과제

- **딥러닝 모델 학습**: 현재 classical CV 기반 → 실제 U-Net/Mask R-CNN/Siamese Net 학습 데이터 구축 및 모델 훈련
- **G-Band 템플릿 DB**: 실제 이디오그램 밴딩 패턴 데이터베이스 구축으로 분류 정확도 향상
- **구조적 이상 검출 강화**: 전좌(translocation), 역위(inversion) 등 구조적 이상의 CV 기반 검출
- **대규모 검증**: 임상 데이터셋으로 민감도/특이도 측정
- **YOLO 통합**: 사전학습된 염색체 객체 탐지 모델 연동 (ultralytics YOLO)
