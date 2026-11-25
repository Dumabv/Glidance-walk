namespace GlidanceSafety

/-!
# Математическая спецификация ядра навигационной безопасности Glidance (Lean 4)
Дискретная модель с фиксированной точкой в миллишкале `[0..1000]` (`1 ед. = 0.001`),
обеспечивающая побитовую идентичность между Python-бэкендом, нативным Rust `cdylib`
и автономным мобильным модулем `wasm32-unknown-unknown`.
-/

/-- Трёхуровневая решётка серьёзности навигационной угрозы (`Fin 3` код: `0=clear < 1=caution < 2=danger`) -/
structure Severity where
  code : Fin 3
  deriving DecidableEq, Repr

def Severity.clear   : Severity := ⟨⟨0, by decide⟩⟩
def Severity.caution : Severity := ⟨⟨1, by decide⟩⟩
def Severity.danger  : Severity := ⟨⟨2, by decide⟩⟩

/-- Монотонное отображение `Severity` в линейный порядок `{0, 1, 2}` -/
def Severity.rank (s : Severity) : Nat :=
  s.code.val

/-- Нестрогий порядок на решётке `Severity` -/
def Severity.le (a b : Severity) : Prop :=
  a.rank ≤ b.rank

/-- Строгий порядок на решётке `Severity` -/
def Severity.lt (a b : Severity) : Prop :=
  a.rank < b.rank

instance : LE Severity := ⟨Severity.le⟩
instance : LT Severity := ⟨Severity.lt⟩

instance (a b : Severity) : Decidable (a ≤ b) :=
  inferInstanceAs (Decidable (a.rank ≤ b.rank))

instance (a b : Severity) : Decidable (a < b) :=
  inferInstanceAs (Decidable (a.rank < b.rank))

/-- Точная верхняя грань (`join` / `max`) двух уровней серьёзности угрозы -/
def Severity.max (a b : Severity) : Severity :=
  if a.rank ≥ b.rank then a else b

/-- Точная верхняя грань четырёх источников угрозы (`Left`, `Center`, `Right`, `Curb`) -/
def maxSeverity4 (s1 s2 s3 s4 : Severity) : Severity :=
  Severity.max (Severity.max s1 s2) (Severity.max s3 s4)

/-- Дискретные команды руления (`Fin 5` C-ABI/WASM код, 1:1 с `SteeringAction` в `schemas.py`) -/
structure SteeringAction where
  code : Fin 5
  deriving DecidableEq, Repr

def SteeringAction.CLEAR      : SteeringAction := ⟨⟨0, by decide⟩⟩
def SteeringAction.VEER_LEFT  : SteeringAction := ⟨⟨1, by decide⟩⟩
def SteeringAction.VEER_RIGHT : SteeringAction := ⟨⟨2, by decide⟩⟩
def SteeringAction.STOP       : SteeringAction := ⟨⟨3, by decide⟩⟩
def SteeringAction.CURB_DROP  : SteeringAction := ⟨⟨4, by decide⟩⟩

/-- Состояние одного пространственного коридора в миллишкале `[0..1000]` -/
structure ZoneState where
  severity      : Severity
  obstacleScore : Nat
  deriving DecidableEq, Repr

/-- Итоговое решение навигационного ядра безопасности -/
structure GuidanceDecision where
  overallSeverity : Severity
  action          : SteeringAction
  deriving DecidableEq, Repr

/-! ### Калибровочные константы геометрической защиты (миллишкала `[0..1000]`) -/

/-- Порог схлопывания динамического размаха `[p2, p98]` на однородной близкой стене (`80‰ = 0.080`) -/
def wallSpreadEps : Nat := 80

/-- Минимальная сырая/верхняя близость при схлопывании контраста на стене (`700‰ = 0.700`) -/
def wallProximityMin : Nat := 700

/-- Порог полного перекрытия верхнего коридора в ближнем поле (`840‰ = 0.840`) -/
def wallBlockageMin : Nat := 840

/-- Гарантированный аварийный балл опасности при срабатывании защиты стены (`1000‰ = 1.000`) -/
def wallDangerScore : Nat := 1000

/-- Минимальная близость опорной поверхности перед истинным бордюром (`350‰ = 0.350`) -/
def curbMinFgFloor : Nat := 350

/-- Максимально допустимое превышение над полом в ближней зоне бордюра (`120‰ = 0.120`):
    блокирует ложное срабатывание `CURB_DROP` на низкие коробки и препятствия -/
def boxMaxExcess : Nat := 120

/-! ### Базовые арифметические и темпоральные операторы миллишкалы -/

/-- Симметричное абсолютное отклонение `|a - b|` в натуральной миллишкале -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Вычитание ожидаемого профиля пола с защитным зазором:
    `max(0, closeness - (floorPrior + margin))` -/
def computeExcess (closeness floorPrior margin : Nat) : Nat :=
  closeness - (floorPrior + margin)

/-- Выпуклый шаг экспоненциального сглаживания (EMA) в миллишкале `[0..1000]`:
    `(alpha * raw + (1000 - alpha) * prev) / 1000` -/
def emaStep (raw prev alpha : Nat) : Nat :=
  (alpha * raw + (1000 - alpha) * prev) / 1000

/-- Обновление счётчика последовательных срабатываний опасности (мгновенный сброс при `false`) -/
def updateStreak (streak : Nat) (triggered : Bool) : Nat :=
  if triggered then streak + 1 else 0

/-- Траектория счётчика серий по потоку кадров -/
def runStreak (initialStreak : Nat) (frames : List Bool) : Nat :=
  frames.foldl updateStreak initialStreak

/-- Предикат подтверждения устойчивой угрозы после `confirmFrames` последовательных кадров -/
def isDangerConfirmed (streak confirmFrames : Nat) : Bool :=
  decide (confirmFrames ≥ 1) && decide (streak ≥ confirmFrames)

/-! ### Предусловия и теоремы обитаемости (защита от Vacuous Truth) -/

/-- Предусловие сценария безопасного обхода центрального препятствия -/
def SafeEscapeScenario (left center right : ZoneState) (curbDetected frameFresh : Bool) : Prop :=
  frameFresh = true ∧
  curbDetected = false ∧
  center.severity = .danger ∧
  (left.severity ≠ .danger ∨ right.severity ≠ .danger)

/-- ЗАЩИТА ОТ ПУСТОЙ ИСТИНЫ 1: сценарий обхода центрального препятствия выполним -/
theorem safeEscapeScenario_satisfiable :
    ∃ (left center right : ZoneState) (curbDetected frameFresh : Bool),
      SafeEscapeScenario left center right curbDetected frameFresh := by
  refine ⟨⟨.clear, 80⟩, ⟨.danger, 720⟩, ⟨.caution, 310⟩, false, true, ?_⟩
  exact ⟨rfl, rfl, rfl, Or.inl (by decide)⟩

/-- Предусловие идеального наклонного плоского пола в пределах допуска `margin` -/
def FlatTiltedFloorPre (closeness floorPrior : Nat → Nat) (margin : Nat) : Prop :=
  0 < margin ∧ margin ≤ 400 ∧
  ∀ y, closeness y ≤ 1000 ∧ floorPrior y ≤ 1000 ∧ absDiff (closeness y) (floorPrior y) ≤ margin

/-- ЗАЩИТА ОТ ПУСТОЙ ИСТИНЫ 2: нетривиальный наклонный профиль плоского пола существует -/
theorem flatTiltedFloorPre_satisfiable :
    ∃ (closeness floorPrior : Nat → Nat) (margin : Nat),
      FlatTiltedFloorPre closeness floorPrior margin := by
  refine ⟨fun y => if y ≤ 600 then 100 + y else 700,
          fun y => if y ≤ 600 then 80 + y else 680,
          130, ?_⟩
  refine ⟨by decide, by decide, ?_⟩
  intro y
  by_cases hy : y ≤ 600 <;> simp [hy, absDiff] <;> omega

/-- Предусловие однородной близкой стены с вырожденным перцентильным контрастом -/
def UniformCloseWallPre (spread upperMedian : Nat) : Prop :=
  spread ≤ wallSpreadEps ∧ upperMedian ≥ wallProximityMin ∧ upperMedian ≤ 1000

/-- ЗАЩИТА ОТ ПУСТОЙ ИСТИНЫ 3: сценарий однородной близкой стены выполним -/
theorem uniformCloseWallPre_satisfiable :
    ∃ spread upperMedian : Nat, UniformCloseWallPre spread upperMedian := by
  refine ⟨35, 820, ?_⟩
  unfold UniformCloseWallPre wallSpreadEps wallProximityMin
  omega

/-- Предусловие истинного бордюра / ступеньки вниз -/
def TrueCurbPre (dropGradient fgFloor fgExcess thresh : Nat) : Prop :=
  40 ≤ thresh ∧ thresh ≤ 600 ∧
  dropGradient ≥ thresh ∧
  fgFloor ≥ curbMinFgFloor ∧
  fgExcess ≤ boxMaxExcess

/-- ЗАЩИТА ОТ ПУСТОЙ ИСТИНЫ 4: сценарий истинного бордюра выполним -/
theorem trueCurbPre_satisfiable :
    ∃ dropGradient fgFloor fgExcess thresh : Nat,
      TrueCurbPre dropGradient fgFloor fgExcess thresh := by
  refine ⟨210, 520, 45, 140, ?_⟩
  unfold TrueCurbPre curbMinFgFloor boxMaxExcess
  omega

/-- Предусловие низкой коробки в ближнем поле с резким перепадом на верхней грани -/
def LowBoxPre (dropGradient fgFloor fgExcess thresh : Nat) : Prop :=
  40 ≤ thresh ∧
  dropGradient ≥ thresh ∧
  fgFloor ≥ curbMinFgFloor ∧
  fgExcess > boxMaxExcess

/-- ЗАЩИТА ОТ ПУСТОЙ ИСТИНЫ 5: сценарий низкой коробки, имитирующей градиент бордюра, выполним -/
theorem lowBoxPre_satisfiable :
    ∃ dropGradient fgFloor fgExcess thresh : Nat,
      LowBoxPre dropGradient fgFloor fgExcess thresh := by
  refine ⟨240, 680, 310, 140, ?_⟩
  unfold LowBoxPre curbMinFgFloor boxMaxExcess
  omega

/-- Предусловие корректно откалиброванного EMA-фильтра и темпорального дебаунса -/
def ValidFilterPre (raw prev alpha confirmFrames : Nat) : Prop :=
  raw ≤ 1000 ∧ prev ≤ 1000 ∧ 0 < alpha ∧ alpha ≤ 1000 ∧ 1 < confirmFrames

/-- ЗАЩИТА ОТ ПУСТОЙ ИСТИНЫ 6: параметры фильтрации Glidance по умолчанию обитаемы -/
theorem validFilterPre_satisfiable :
    ∃ raw prev alpha confirmFrames : Nat,
      ValidFilterPre raw prev alpha confirmFrames := by
  refine ⟨620, 180, 550, 2, ?_⟩
  unfold ValidFilterPre
  omega

/-! ### Целевые функции ядра безопасности (реализуются в `GlidanceSafety.Solution.lean`) -/

/-- Верифицированная функция принятия решения о рулении и уровне тревоги -/
def decideGuidance
    (left center right : ZoneState)
    (curbDetected : Bool)
    (frameFresh : Bool) : GuidanceDecision := sorry

/-- Детектор вырождения глубины на однородной близкой стене или полном перекрытии коридора -/
def wallOverride (spread upperMedian : Nat) : Bool := sorry

/-- Комбинированная оценка опасности коридора с защитой от вычитания стены -/
def combineWallHazard (relativeExcessScore spread upperMedian : Nat) : Nat := sorry

/-- Дискриминатор истинного бордюра вниз (с защитой от низких препятствий/коробок) -/
def isTrueCurbDrop (dropGradient fgFloor fgExcess thresh : Nat) : Bool := sorry

/-- Асимметричный классификатор зоны с гистерезисом подтверждения серии кадров -/
def classifyZone (score cautionThresh dangerThresh prevStreak confirmFrames : Nat) : Severity × Nat := sorry

/-! ### Раздел 1. Решётка серьёзности и инварианты безопасности `decideGuidance` -/

/-- Теорема 1 (Fail-Closed Telemetry Safety):
    при устаревшем кадре или обрыве связи система обязана выдать `STOP` и `danger` -/
theorem decideGuidance_failClosed
    (left center right : ZoneState) (curbDetected : Bool) (frameFresh : Bool)
    (hStale : frameFresh = false) :
    (decideGuidance left center right curbDetected frameFresh).action = .STOP ∧
    (decideGuidance left center right curbDetected frameFresh).overallSeverity = .danger := sorry

/-- Теорема 2 (Strict Severity Dominance — устраняет баг занижения угрозы в `hazard_detector.py:255`):
    итоговая серьёзность никогда не ниже максимума по всем трём зонам и детектору бордюра -/
theorem decideGuidance_severity_dominance
    (left center right : ZoneState) (curbDetected frameFresh : Bool) :
    maxSeverity4
      left.severity
      center.severity
      right.severity
      (if curbDetected then .danger else .clear) ≤
    (decideGuidance left center right curbDetected frameFresh).overallSeverity := sorry

/-- Теорема 2b (Exact Fresh Dominance):
    при свежем кадре итоговая серьёзность в точности равна точной верхней грани источников -/
theorem decideGuidance_severity_exact
    (left center right : ZoneState) (curbDetected : Bool) :
    (decideGuidance left center right curbDetected true).overallSeverity =
    maxSeverity4
      left.severity
      center.severity
      right.severity
      (if curbDetected then .danger else .clear) := sorry

/-- Теорема 3a (Never Steer Left Into Danger):
    команда `VEER_LEFT` невозможна, если левый коридор находится в состоянии `danger` -/
theorem decideGuidance_never_veer_left_danger
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hAct : (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT) :
    left.severity ≠ .danger := sorry

/-- Теорема 3b (Never Steer Right Into Danger):
    команда `VEER_RIGHT` невозможна, если правый коридор находится в состоянии `danger` -/
theorem decideGuidance_never_veer_right_danger
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hAct : (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT) :
    right.severity ≠ .danger := sorry

/-- Теорема 3c (Total Blockage Must Stop):
    если все три коридора в `danger` и бордюр не активен, единственное допустимое действие — `STOP` -/
theorem decideGuidance_all_danger_stop
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hCenter : center.severity = .danger)
    (hLeft   : left.severity = .danger)
    (hRight  : right.severity = .danger)
    (hNoCurb : ¬curbDetected) :
    (decideGuidance left center right curbDetected frameFresh).action = .STOP := sorry

/-- Теорема 3d (Curb Drop Priority):
    при свежем кадре и подтверждённом бордюре выдаётся `CURB_DROP` с уровнем `danger` -/
theorem decideGuidance_curb_priority
    (left center right : ZoneState) :
    (decideGuidance left center right true true).action = .CURB_DROP ∧
    (decideGuidance left center right true true).overallSeverity = .danger := sorry

/-- Теорема 4a (Optimal Safe Escape — Non-Freezing):
    если центр заблокирован (`danger`), но хотя бы один боковой коридор безопасен (`≠ danger`),
    ядро выбирает манёвр обхода (`VEER_LEFT` или `VEER_RIGHT`) и никогда не зависает в `STOP` -/
theorem decideGuidance_escape_exists
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hEsc : SafeEscapeScenario left center right curbDetected frameFresh) :
    (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT ∨
    (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT := sorry

/-- Теорема 4b (Optimal Safe Escape — Strict Severity Preference):
    при обходе центрального препятствия ядро строго предпочитает коридор с меньшей серьёзностью -/
theorem decideGuidance_escape_prefers_lower_severity
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hEsc : SafeEscapeScenario left center right curbDetected frameFresh) :
    (left.severity < right.severity →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT) ∧
    (right.severity < left.severity →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT) := sorry

/-- Теорема 4c (Optimal Safe Escape — Lower Obstacle Score Tie-Breaking):
    при равной серьёзности безопасных боковых коридоров выбирается сторона с меньшим `obstacleScore` -/
theorem decideGuidance_escape_prefers_lower_score
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hEsc : SafeEscapeScenario left center right curbDetected frameFresh)
    (hEqSev : left.severity = right.severity) :
    (left.obstacleScore ≤ right.obstacleScore →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT) ∧
    (right.obstacleScore < left.obstacleScore →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT) := sorry

/-- Теорема 4d (Clear Corridor Passthrough):
    при полностью свободных коридорах, отсутствии бордюра и свежем кадре выдаётся `CLEAR` и `clear` -/
theorem decideGuidance_clear_corridor
    (left center right : ZoneState)
    (hLeft   : left.severity = .clear)
    (hCenter : center.severity = .clear)
    (hRight  : right.severity = .clear) :
    (decideGuidance left center right false true).action = .CLEAR ∧
    (decideGuidance left center right false true).overallSeverity = .clear := sorry

/-- Теорема 4e (Symmetric Caution Corridor Stability — устраняет ложный увод вправо `CE1C`):
    если все три коридора в `caution` с близкими баллами (`≤ 50‰`), ядро удерживает курс `CLEAR` с уровнем `caution` -/
theorem decideGuidance_symmetric_caution_centered
    (left center right : ZoneState)
    (hLeft   : left.severity = .caution)
    (hCenter : center.severity = .caution)
    (hRight  : right.severity = .caution)
    (hSym    : absDiff left.obstacleScore right.obstacleScore ≤ 50) :
    (decideGuidance left center right false true).action = .CLEAR ∧
    (decideGuidance left center right false true).overallSeverity = .caution := sorry

/-! ### Раздел 2. Вычитание плоскости пола, защита стены и разделение бордюр / коробка -/

/-- Теорема 5a (Zero False Alarms on Bounded Flat Floor — поточечно):
    если отклонение от ожидаемого профиля пола не превышает `margin`, избыток `computeExcess` равен `0` -/
theorem computeExcess_flat_zero
    (closeness floorPrior margin : Nat)
    (hFlat : absDiff closeness floorPrior ≤ margin) :
    computeExcess closeness floorPrior margin = 0 := sorry

/-- Теорема 5b (Zero False Alarms on Tilted Flat Floor Profile — по всей высоте кадра `∀ y`):
    на любом наклонном плоском полу в полосе допуска избыточная карта препятствий тождественно равна `0` -/
theorem flatFloor_profile_zero_excess
    (closeness floorPrior : Nat → Nat) (margin : Nat)
    (hPre : FlatTiltedFloorPre closeness floorPrior margin) :
    ∀ y, computeExcess (closeness y) (floorPrior y) margin = 0 := sorry

/-- Теорема 5c (Obstacle Sensitivity):
    любой объект, выступающий ближе `floorPrior + margin`, порождает строго положительный `computeExcess` -/
theorem computeExcess_protrusion_pos
    (closeness floorPrior margin : Nat)
    (hObj : closeness > floorPrior + margin) :
    computeExcess closeness floorPrior margin > 0 := sorry

/-- Теорема 6a (Wall-Collapse Guard — однородная близкая стена):
    даже при нулевом относительном контрасте (`relativeExcessScore = 0`), если размах `spread ≤ wallSpreadEps`
    и близость `upperMedian ≥ wallProximityMin`, срабатывает аварийная защита стены -/
theorem wallOverride_uniform_close_wall
    (relativeExcessScore spread upperMedian : Nat)
    (hWall : UniformCloseWallPre spread upperMedian) :
    wallOverride spread upperMedian = true ∧
    combineWallHazard relativeExcessScore spread upperMedian = wallDangerScore := sorry

/-- Теорема 6b (Wall-Collapse Guard — полное перекрытие коридора в ближнем поле):
    при `upperMedian ≥ wallBlockageMin` аварийная защита срабатывает независимо от размаха `spread` -/
theorem wallOverride_full_corridor_blockage
    (relativeExcessScore spread upperMedian : Nat)
    (hBlock : upperMedian ≥ wallBlockageMin) :
    wallOverride spread upperMedian = true ∧
    combineWallHazard relativeExcessScore spread upperMedian = wallDangerScore := sorry

/-- Теорема 6c (Wall Guard Non-Interference on Normal Floor):
    при нормальном перспективном размахе и свободном верхе коридора защита стены не создаёт ложных тревог -/
theorem wallOverride_no_false_alarm_on_floor
    (spread upperMedian : Nat)
    (hSpread : spread > wallSpreadEps)
    (hUpper  : upperMedian < wallBlockageMin) :
    wallOverride spread upperMedian = false ∧
    combineWallHazard 0 spread upperMedian = 0 := sorry

/-- Теорема 6d (Bounded Combined Wall Hazard):
    итоговая оценка коридора всегда остаётся в миллишкале `[0..1000]` -/
theorem combineWallHazard_bounded
    (relativeExcessScore spread upperMedian : Nat) :
    combineWallHazard relativeExcessScore spread upperMedian ≤ 1000 := sorry

/-- Теорема 7a (Curb-Obstacle Mutual Exclusion — устраняет баг низкой коробки в `_detect_curb`):
    выступающий объект или низкая коробка (`fgExcess > boxMaxExcess`) НИКОГДА не классифицируется как бордюр вниз -/
theorem curb_obstacle_mutual_exclusion
    (dropGradient fgFloor fgExcess thresh : Nat)
    (hBox : fgExcess > boxMaxExcess) :
    isTrueCurbDrop dropGradient fgFloor fgExcess thresh = false := sorry

/-- Теорема 7b (True Curb Detection Completeness):
    истинный перепад вниз перед валидным плоским полом гарантированно распознаётся как бордюр -/
theorem isTrueCurbDrop_complete
    (dropGradient fgFloor fgExcess thresh : Nat)
    (hCurb : TrueCurbPre dropGradient fgFloor fgExcess thresh) :
    isTrueCurbDrop dropGradient fgFloor fgExcess thresh = true := sorry

/-- Теорема 7c (Curb Floor & Gradient Gate Soundness):
    `isTrueCurbDrop = true` влечёт выполнение всех трёх физических условий одновременно -/
theorem isTrueCurbDrop_sound
    (dropGradient fgFloor fgExcess thresh : Nat)
    (hDet : isTrueCurbDrop dropGradient fgFloor fgExcess thresh = true) :
    dropGradient ≥ thresh ∧ fgFloor ≥ curbMinFgFloor ∧ fgExcess ≤ boxMaxExcess := sorry

/-! ### Раздел 3. Ограниченность EMA-фильтра и асимметричный темпоральный гистерезис -/

/-- Теорема 8a (EMA Milliscale Boundedness):
    при входах и коэффициенте `alpha` в `[0, 1000]` выход выпуклого EMA остаётся в `[0, 1000]` -/
theorem emaStep_bounded
    (raw prev alpha : Nat)
    (hRaw   : raw ≤ 1000)
    (hPrev  : prev ≤ 1000)
    (hAlpha : alpha ≤ 1000) :
    emaStep raw prev alpha ≤ 1000 := sorry

/-- Теорема 8b (EMA Convex Hull Invariant):
    выход `emaStep` всегда заключён между `min(raw, prev)` и `max(raw, prev)` без перерегулирования -/
theorem emaStep_convex
    (raw prev alpha : Nat)
    (hAlpha : alpha ≤ 1000) :
    Nat.min raw prev ≤ emaStep raw prev alpha ∧
    emaStep raw prev alpha ≤ Nat.max raw prev := sorry

/-- Теорема 9a (Single-Frame Transient Spike Rejection):
    при `1 < confirmFrames` одиночный шумовой выброс из чистого состояния не вызывает подтверждения `danger` -/
theorem singleSpike_not_confirmed
    (confirmFrames : Nat)
    (hConfirm : 1 < confirmFrames) :
    isDangerConfirmed (runStreak 0 [true]) confirmFrames = false ∧
    isDangerConfirmed (runStreak 0 [true, false]) confirmFrames = false := sorry

/-- Теорема 9b (Sub-Window Burst Rejection):
    любая пачка кадров короче `confirmFrames` из нулевого состояния не может подтвердить `danger` -/
theorem shortBurst_not_confirmed
    (frames : List Bool) (confirmFrames : Nat)
    (hShort : frames.length < confirmFrames) :
    isDangerConfirmed (runStreak 0 frames) confirmFrames = false := sorry

/-- Теорема 9c (Deterministic Multi-Frame Hazard Confirmation):
    ровно `confirmFrames` последовательных кадров угрозы детерминированно подтверждают `danger` -/
theorem consecutiveHazard_confirms
    (initialStreak confirmFrames : Nat)
    (hPos : 1 ≤ confirmFrames) :
    isDangerConfirmed (runStreak initialStreak (List.replicate confirmFrames true)) confirmFrames = true := sorry

/-- Теорема 9d (Asymmetric Zone Classifier Hysteresis):
    одиночный пик `score ≥ dangerThresh` при `1 < confirmFrames` переводит зону только в `caution` (не `danger`),
    а достижение `confirmFrames` последовательных кадров гарантированно переводит зону в `danger` -/
theorem classifyZone_hysteresis
    (score cautionThresh dangerThresh confirmFrames : Nat)
    (hConfirm : 1 < confirmFrames)
    (hHigh    : score ≥ dangerThresh) :
    (classifyZone score cautionThresh dangerThresh 0 confirmFrames).1 = .caution ∧
    (classifyZone score cautionThresh dangerThresh (confirmFrames - 1) confirmFrames).1 = .danger := sorry

end GlidanceSafety
