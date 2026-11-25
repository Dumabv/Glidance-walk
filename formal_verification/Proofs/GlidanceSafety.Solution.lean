namespace GlidanceSafety

/-!
# Полная формальная реализация и доказательства ядра безопасности Glidance (Lean 4)
0 `sorryAx`, 0 пользовательских аксиом, 0 `unsafe`/`partial`.
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

/-! ### Реализация целевых функций ядра безопасности -/

/-- Лексикографический выбор более безопасного бокового коридора:
    1) Строго меньшая серьёзность угрозы (`Severity.rank`)
    2) При равной серьёзности — меньший балл препятствия (`obstacleScore`) -/
def pickSaferSide (left right : ZoneState) : SteeringAction :=
  if left.severity.rank < right.severity.rank then
    .VEER_LEFT
  else if right.severity.rank < left.severity.rank then
    .VEER_RIGHT
  else if left.obstacleScore ≤ right.obstacleScore then
    .VEER_LEFT
  else
    .VEER_RIGHT

/-- Верифицированная функция принятия решения о рулении и уровне тревоги -/
def decideGuidance
    (left center right : ZoneState)
    (curbDetected : Bool)
    (frameFresh : Bool) : GuidanceDecision :=
  if !frameFresh then
    { overallSeverity := .danger, action := .STOP }
  else
    let overall := maxSeverity4
      left.severity
      center.severity
      right.severity
      (if curbDetected then Severity.danger else Severity.clear)
    let act : SteeringAction :=
      if curbDetected then
        .CURB_DROP
      else if decide (left.severity = .danger) && decide (right.severity = .danger) then
        .STOP
      else if decide (center.severity = .danger) then
        pickSaferSide left right
      else if decide (center.severity = .caution) then
        if decide (left.severity = .caution) && decide (right.severity = .caution) &&
           decide (absDiff left.obstacleScore right.obstacleScore ≤ 50) then
          .CLEAR
        else
          pickSaferSide left right
      else
        if decide (left.severity = .danger) then
          .VEER_RIGHT
        else if decide (right.severity = .danger) then
          .VEER_LEFT
        else if decide (left.severity = .caution) && decide (right.severity = .clear) then
          .VEER_RIGHT
        else if decide (right.severity = .caution) && decide (left.severity = .clear) then
          .VEER_LEFT
        else
          .CLEAR
    { overallSeverity := overall, action := act }

/-- Детектор вырождения глубины на однородной близкой стене или полном перекрытии коридора -/
def wallOverride (spread upperMedian : Nat) : Bool :=
  (decide (spread ≤ wallSpreadEps) && decide (upperMedian ≥ wallProximityMin)) ||
  decide (upperMedian ≥ wallBlockageMin)

/-- Комбинированная оценка опасности коридора с защитой от вычитания стены -/
def combineWallHazard (relativeExcessScore spread upperMedian : Nat) : Nat :=
  if wallOverride spread upperMedian then
    wallDangerScore
  else
    Nat.min relativeExcessScore 1000

/-- Дискриминатор истинного бордюра вниз (с защитой от низких препятствий/коробок) -/
def isTrueCurbDrop (dropGradient fgFloor fgExcess thresh : Nat) : Bool :=
  decide (dropGradient ≥ thresh) &&
  decide (fgFloor ≥ curbMinFgFloor) &&
  decide (fgExcess ≤ boxMaxExcess)

/-- Асимметричный классификатор зоны с гистерезисом подтверждения серии кадров -/
def classifyZone (score cautionThresh dangerThresh prevStreak confirmFrames : Nat) : Severity × Nat :=
  if score ≥ dangerThresh then
    let nextStreak := prevStreak + 1
    if nextStreak ≥ confirmFrames then
      (Severity.danger, nextStreak)
    else
      (Severity.caution, nextStreak)
  else if score ≥ cautionThresh then
    (Severity.caution, 0)
  else
    (Severity.clear, 0)

/-! ### Вспомогательные леммы для решётки `Severity` и `pickSaferSide` -/

private theorem severity_cases (s : Severity) :
    s = .clear ∨ s = .caution ∨ s = .danger := by
  rcases s with ⟨⟨v, hv⟩⟩
  match v with
  | 0 => exact Or.inl rfl
  | 1 => exact Or.inr (Or.inl rfl)
  | 2 => exact Or.inr (Or.inr rfl)
  | n + 3 => omega

private theorem severity_le_danger (s : Severity) : s ≤ .danger := by
  rcases severity_cases s with rfl | rfl | rfl <;> decide

private theorem maxSeverity4_fourth_danger (s1 s2 s3 : Severity) :
    maxSeverity4 s1 s2 s3 .danger = .danger := by
  rcases severity_cases s1 with rfl | rfl | rfl <;>
    rcases severity_cases s2 with rfl | rfl | rfl <;>
    rcases severity_cases s3 with rfl | rfl | rfl <;> rfl

private theorem severity_le_refl (s : Severity) : s ≤ s :=
  Nat.le_refl s.rank

private theorem pickSaferSide_veer_left_or_right (left right : ZoneState) :
    pickSaferSide left right = .VEER_LEFT ∨ pickSaferSide left right = .VEER_RIGHT := by
  unfold pickSaferSide
  split
  · exact Or.inl rfl
  · split
    · exact Or.inr rfl
    · split
      · exact Or.inl rfl
      · exact Or.inr rfl

private theorem pickSaferSide_not_danger_left
    (left right : ZoneState)
    (hNotBoth : ¬(left.severity = .danger ∧ right.severity = .danger))
    (hAct : pickSaferSide left right = .VEER_LEFT) :
    left.severity ≠ .danger := by
  rcases left with ⟨ls, lscore⟩
  rcases right with ⟨rs, rscore⟩
  rcases severity_cases ls with rfl | rfl | rfl <;>
    rcases severity_cases rs with rfl | rfl | rfl <;>
    by_cases hsc : lscore ≤ rscore <;>
    simp_all [pickSaferSide, Severity.rank, Severity.clear, Severity.caution, Severity.danger,
      SteeringAction.VEER_LEFT, SteeringAction.VEER_RIGHT]

private theorem pickSaferSide_not_danger_right
    (left right : ZoneState)
    (hNotBoth : ¬(left.severity = .danger ∧ right.severity = .danger))
    (hAct : pickSaferSide left right = .VEER_RIGHT) :
    right.severity ≠ .danger := by
  rcases left with ⟨ls, lscore⟩
  rcases right with ⟨rs, rscore⟩
  rcases severity_cases ls with rfl | rfl | rfl <;>
    rcases severity_cases rs with rfl | rfl | rfl <;>
    by_cases hsc : lscore ≤ rscore <;>
    simp_all [pickSaferSide, Severity.rank, Severity.clear, Severity.caution, Severity.danger,
      SteeringAction.VEER_LEFT, SteeringAction.VEER_RIGHT]

/-! ### Раздел 1. Доказательства теорем `decideGuidance` -/

/-- Теорема 1 (Fail-Closed Telemetry Safety) -/
theorem decideGuidance_failClosed
    (left center right : ZoneState) (curbDetected : Bool) (frameFresh : Bool)
    (hStale : frameFresh = false) :
    (decideGuidance left center right curbDetected frameFresh).action = .STOP ∧
    (decideGuidance left center right curbDetected frameFresh).overallSeverity = .danger := by
  subst hStale
  simp [decideGuidance]

/-- Теорема 2 (Strict Severity Dominance) -/
theorem decideGuidance_severity_dominance
    (left center right : ZoneState) (curbDetected frameFresh : Bool) :
    maxSeverity4
      left.severity
      center.severity
      right.severity
      (if curbDetected then .danger else .clear) ≤
    (decideGuidance left center right curbDetected frameFresh).overallSeverity := by
  cases frameFresh with
  | false =>
    simp only [decideGuidance, Bool.not_false, ↓reduceIte]
    exact severity_le_danger _
  | true =>
    exact severity_le_refl _

/-- Теорема 2b (Exact Fresh Dominance) -/
theorem decideGuidance_severity_exact
    (left center right : ZoneState) (curbDetected : Bool) :
    (decideGuidance left center right curbDetected true).overallSeverity =
    maxSeverity4
      left.severity
      center.severity
      right.severity
      (if curbDetected then .danger else .clear) := by
  rfl

/-- Теорема 3a (Never Steer Left Into Danger) -/
theorem decideGuidance_never_veer_left_danger
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hAct : (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT) :
    left.severity ≠ .danger := by
  cases frameFresh with
  | false =>
    simp [decideGuidance, SteeringAction.STOP, SteeringAction.VEER_LEFT] at hAct
  | true =>
    cases curbDetected with
    | true =>
      simp [decideGuidance, SteeringAction.CURB_DROP, SteeringAction.VEER_LEFT] at hAct
    | false =>
      by_cases hBoth : left.severity = .danger ∧ right.severity = .danger
      · rcases hBoth with ⟨hL, hR⟩
        simp [decideGuidance, hL, hR, SteeringAction.STOP, SteeringAction.VEER_LEFT] at hAct
      · have hBothFalse : (decide (left.severity = .danger) && decide (right.severity = .danger)) = false := by
          by_cases h1 : left.severity = .danger <;> by_cases h2 : right.severity = .danger <;> simp_all
        by_cases hCD : center.severity = .danger
        · simp [decideGuidance, hBothFalse, hCD] at hAct
          exact pickSaferSide_not_danger_left left right hBoth hAct
        · by_cases hCC : center.severity = .caution
          · have hCD' : decide (center.severity = .danger) = false := decide_eq_false hCD
            have hCC' : decide (center.severity = .caution) = true := decide_eq_true hCC
            simp only [decideGuidance, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hBothFalse, hCD', hCC'] at hAct
            split at hAct
            · exfalso; revert hAct; decide
            · exact pickSaferSide_not_danger_left left right hBoth hAct
          · rcases left with ⟨ls, lscore⟩
            rcases right with ⟨rs, rscore⟩
            rcases severity_cases ls with rfl | rfl | rfl <;>
              rcases severity_cases rs with rfl | rfl | rfl <;>
              simp_all [decideGuidance, Severity.clear, Severity.caution, Severity.danger,
                SteeringAction.CLEAR, SteeringAction.VEER_LEFT, SteeringAction.VEER_RIGHT, SteeringAction.STOP]

/-- Теорема 3b (Never Steer Right Into Danger) -/
theorem decideGuidance_never_veer_right_danger
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hAct : (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT) :
    right.severity ≠ .danger := by
  cases frameFresh with
  | false =>
    simp [decideGuidance, SteeringAction.STOP, SteeringAction.VEER_RIGHT] at hAct
  | true =>
    cases curbDetected with
    | true =>
      simp [decideGuidance, SteeringAction.CURB_DROP, SteeringAction.VEER_RIGHT] at hAct
    | false =>
      by_cases hBoth : left.severity = .danger ∧ right.severity = .danger
      · rcases hBoth with ⟨hL, hR⟩
        simp [decideGuidance, hL, hR, SteeringAction.STOP, SteeringAction.VEER_RIGHT] at hAct
      · have hBothFalse : (decide (left.severity = .danger) && decide (right.severity = .danger)) = false := by
          by_cases h1 : left.severity = .danger <;> by_cases h2 : right.severity = .danger <;> simp_all
        by_cases hCD : center.severity = .danger
        · simp [decideGuidance, hBothFalse, hCD] at hAct
          exact pickSaferSide_not_danger_right left right hBoth hAct
        · by_cases hCC : center.severity = .caution
          · have hCD' : decide (center.severity = .danger) = false := decide_eq_false hCD
            have hCC' : decide (center.severity = .caution) = true := decide_eq_true hCC
            simp only [decideGuidance, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hBothFalse, hCD', hCC'] at hAct
            split at hAct
            · exfalso; revert hAct; decide
            · exact pickSaferSide_not_danger_right left right hBoth hAct
          · rcases left with ⟨ls, lscore⟩
            rcases right with ⟨rs, rscore⟩
            rcases severity_cases ls with rfl | rfl | rfl <;>
              rcases severity_cases rs with rfl | rfl | rfl <;>
              simp_all [decideGuidance, Severity.clear, Severity.caution, Severity.danger,
                SteeringAction.CLEAR, SteeringAction.VEER_LEFT, SteeringAction.VEER_RIGHT, SteeringAction.STOP]

/-- Теорема 3c (Total Blockage Must Stop) -/
theorem decideGuidance_all_danger_stop
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hCenter : center.severity = .danger)
    (hLeft   : left.severity = .danger)
    (hRight  : right.severity = .danger)
    (hNoCurb : ¬curbDetected) :
    (decideGuidance left center right curbDetected frameFresh).action = .STOP := by
  have hCurbFalse : curbDetected = false := Bool.eq_false_iff.mpr hNoCurb
  subst hCurbFalse
  cases frameFresh <;> simp [decideGuidance, hLeft, hRight, hCenter]

/-- Теорема 3d (Curb Drop Priority) -/
theorem decideGuidance_curb_priority
    (left center right : ZoneState) :
    (decideGuidance left center right true true).action = .CURB_DROP ∧
    (decideGuidance left center right true true).overallSeverity = .danger := by
  simp [decideGuidance, maxSeverity4_fourth_danger]

/-- Теорема 4a (Optimal Safe Escape — Non-Freezing) -/
theorem decideGuidance_escape_exists
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hEsc : SafeEscapeScenario left center right curbDetected frameFresh) :
    (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT ∨
    (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT := by
  rcases hEsc with ⟨hFresh, hCurb, hCenter, hSafeSide⟩
  subst hFresh hCurb
  have hBothFalse : (decide (left.severity = .danger) && decide (right.severity = .danger)) = false := by
    by_cases h1 : left.severity = .danger <;> by_cases h2 : right.severity = .danger <;> simp_all
  simp only [decideGuidance, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hBothFalse, hCenter, decide_true]
  exact pickSaferSide_veer_left_or_right left right

/-- Теорема 4b (Optimal Safe Escape — Strict Severity Preference) -/
theorem decideGuidance_escape_prefers_lower_severity
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hEsc : SafeEscapeScenario left center right curbDetected frameFresh) :
    (left.severity < right.severity →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT) ∧
    (right.severity < left.severity →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT) := by
  rcases hEsc with ⟨hFresh, hCurb, hCenter, hSafeSide⟩
  subst hFresh hCurb
  have hBothFalse : (decide (left.severity = .danger) && decide (right.severity = .danger)) = false := by
    by_cases h1 : left.severity = .danger <;> by_cases h2 : right.severity = .danger <;> simp_all
  simp only [decideGuidance, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hBothFalse, hCenter, decide_true]
  constructor
  · intro hLt
    have hRankLt : left.severity.rank < right.severity.rank := hLt
    simp [pickSaferSide, hRankLt]
  · intro hRt
    have hRankRt : right.severity.rank < left.severity.rank := hRt
    have hNotLt : ¬(left.severity.rank < right.severity.rank) := by omega
    simp [pickSaferSide, hNotLt, hRankRt]

/-- Теорема 4c (Optimal Safe Escape — Lower Obstacle Score Tie-Breaking) -/
theorem decideGuidance_escape_prefers_lower_score
    (left center right : ZoneState) (curbDetected frameFresh : Bool)
    (hEsc : SafeEscapeScenario left center right curbDetected frameFresh)
    (hEqSev : left.severity = right.severity) :
    (left.obstacleScore ≤ right.obstacleScore →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_LEFT) ∧
    (right.obstacleScore < left.obstacleScore →
      (decideGuidance left center right curbDetected frameFresh).action = .VEER_RIGHT) := by
  rcases hEsc with ⟨hFresh, hCurb, hCenter, hSafeSide⟩
  subst hFresh hCurb
  have hBothFalse : (decide (left.severity = .danger) && decide (right.severity = .danger)) = false := by
    by_cases h1 : left.severity = .danger <;> by_cases h2 : right.severity = .danger <;> simp_all
  simp only [decideGuidance, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hBothFalse, hCenter, decide_true]
  have hNotLt1 : ¬(left.severity.rank < right.severity.rank) := by rw [hEqSev]; omega
  have hNotLt2 : ¬(right.severity.rank < left.severity.rank) := by rw [hEqSev]; omega
  constructor
  · intro hLe
    simp [pickSaferSide, hNotLt1, hNotLt2, hLe]
  · intro hGt
    have hNotLe : ¬(left.obstacleScore ≤ right.obstacleScore) := by omega
    simp [pickSaferSide, hNotLt1, hNotLt2, hNotLe]

/-- Теорема 4d (Clear Corridor Passthrough) -/
theorem decideGuidance_clear_corridor
    (left center right : ZoneState)
    (hLeft   : left.severity = .clear)
    (hCenter : center.severity = .clear)
    (hRight  : right.severity = .clear) :
    (decideGuidance left center right false true).action = .CLEAR ∧
    (decideGuidance left center right false true).overallSeverity = .clear := by
  simp [decideGuidance, hLeft, hCenter, hRight,
    maxSeverity4, Severity.max, Severity.rank, Severity.clear, Severity.caution, Severity.danger]

/-- Теорема 4e (Symmetric Caution Corridor Stability) -/
theorem decideGuidance_symmetric_caution_centered
    (left center right : ZoneState)
    (hLeft   : left.severity = .caution)
    (hCenter : center.severity = .caution)
    (hRight  : right.severity = .caution)
    (hSym    : absDiff left.obstacleScore right.obstacleScore ≤ 50) :
    (decideGuidance left center right false true).action = .CLEAR ∧
    (decideGuidance left center right false true).overallSeverity = .caution := by
  simp [decideGuidance, hLeft, hCenter, hRight, hSym,
    maxSeverity4, Severity.max, Severity.rank, Severity.clear, Severity.caution, Severity.danger]

/-! ### Раздел 2. Доказательства вычитания плоскости пола, защиты стены и бордюра -/

/-- Теорема 5a (Zero False Alarms on Bounded Flat Floor — поточечно) -/
theorem computeExcess_flat_zero
    (closeness floorPrior margin : Nat)
    (hFlat : absDiff closeness floorPrior ≤ margin) :
    computeExcess closeness floorPrior margin = 0 := by
  unfold computeExcess
  unfold absDiff at hFlat
  split at hFlat <;> omega

/-- Теорема 5b (Zero False Alarms on Tilted Flat Floor Profile — `∀ y`) -/
theorem flatFloor_profile_zero_excess
    (closeness floorPrior : Nat → Nat) (margin : Nat)
    (hPre : FlatTiltedFloorPre closeness floorPrior margin) :
    ∀ y, computeExcess (closeness y) (floorPrior y) margin = 0 := by
  intro y
  exact computeExcess_flat_zero (closeness y) (floorPrior y) margin (hPre.2.2 y).2.2

/-- Теорема 5c (Obstacle Sensitivity) -/
theorem computeExcess_protrusion_pos
    (closeness floorPrior margin : Nat)
    (hObj : closeness > floorPrior + margin) :
    computeExcess closeness floorPrior margin > 0 := by
  unfold computeExcess
  omega

/-- Теорема 6a (Wall-Collapse Guard — однородная близкая стена) -/
theorem wallOverride_uniform_close_wall
    (relativeExcessScore spread upperMedian : Nat)
    (hWall : UniformCloseWallPre spread upperMedian) :
    wallOverride spread upperMedian = true ∧
    combineWallHazard relativeExcessScore spread upperMedian = wallDangerScore := by
  rcases hWall with ⟨hSpread, hProx, _⟩
  have hOv : wallOverride spread upperMedian = true := by
    unfold wallOverride
    simp [hSpread, hProx]
  refine ⟨hOv, ?_⟩
  unfold combineWallHazard
  simp [hOv]

/-- Теорема 6b (Wall-Collapse Guard — полное перекрытие коридора в ближнем поле) -/
theorem wallOverride_full_corridor_blockage
    (relativeExcessScore spread upperMedian : Nat)
    (hBlock : upperMedian ≥ wallBlockageMin) :
    wallOverride spread upperMedian = true ∧
    combineWallHazard relativeExcessScore spread upperMedian = wallDangerScore := by
  have hOv : wallOverride spread upperMedian = true := by
    unfold wallOverride
    simp [hBlock]
  refine ⟨hOv, ?_⟩
  unfold combineWallHazard
  simp [hOv]

/-- Теорема 6c (Wall Guard Non-Interference on Normal Floor) -/
theorem wallOverride_no_false_alarm_on_floor
    (spread upperMedian : Nat)
    (hSpread : spread > wallSpreadEps)
    (hUpper  : upperMedian < wallBlockageMin) :
    wallOverride spread upperMedian = false ∧
    combineWallHazard 0 spread upperMedian = 0 := by
  have hNotSpread : ¬(spread ≤ wallSpreadEps) := by omega
  have hNotBlock  : ¬(upperMedian ≥ wallBlockageMin) := by omega
  have hOv : wallOverride spread upperMedian = false := by
    unfold wallOverride
    simp [hNotSpread, hNotBlock]
  refine ⟨hOv, ?_⟩
  unfold combineWallHazard
  simp [hOv]

/-- Теорема 6d (Bounded Combined Wall Hazard) -/
theorem combineWallHazard_bounded
    (relativeExcessScore spread upperMedian : Nat) :
    combineWallHazard relativeExcessScore spread upperMedian ≤ 1000 := by
  unfold combineWallHazard wallDangerScore
  split
  · decide
  · exact Nat.min_le_right relativeExcessScore 1000

/-- Теорема 7a (Curb-Obstacle Mutual Exclusion) -/
theorem curb_obstacle_mutual_exclusion
    (dropGradient fgFloor fgExcess thresh : Nat)
    (hBox : fgExcess > boxMaxExcess) :
    isTrueCurbDrop dropGradient fgFloor fgExcess thresh = false := by
  have hNotEx : ¬(fgExcess ≤ boxMaxExcess) := by omega
  unfold isTrueCurbDrop
  simp [hNotEx]

/-- Теорема 7b (True Curb Detection Completeness) -/
theorem isTrueCurbDrop_complete
    (dropGradient fgFloor fgExcess thresh : Nat)
    (hCurb : TrueCurbPre dropGradient fgFloor fgExcess thresh) :
    isTrueCurbDrop dropGradient fgFloor fgExcess thresh = true := by
  rcases hCurb with ⟨_, _, hGrad, hFloor, hEx⟩
  unfold isTrueCurbDrop
  simp [hGrad, hFloor, hEx]

/-- Теорема 7c (Curb Floor & Gradient Gate Soundness) -/
theorem isTrueCurbDrop_sound
    (dropGradient fgFloor fgExcess thresh : Nat)
    (hDet : isTrueCurbDrop dropGradient fgFloor fgExcess thresh = true) :
    dropGradient ≥ thresh ∧ fgFloor ≥ curbMinFgFloor ∧ fgExcess ≤ boxMaxExcess := by
  unfold isTrueCurbDrop at hDet
  by_cases hGrad : dropGradient ≥ thresh
  · by_cases hFloor : fgFloor ≥ curbMinFgFloor
    · by_cases hEx : fgExcess ≤ boxMaxExcess
      · exact ⟨hGrad, hFloor, hEx⟩
      · simp [hGrad, hFloor, hEx] at hDet
    · simp [hGrad, hFloor] at hDet
  · simp [hGrad] at hDet

/-! ### Раздел 3. Доказательства EMA-фильтра и темпорального гистерезиса -/

/-- Теорема 8a (EMA Milliscale Boundedness) -/
theorem emaStep_bounded
    (raw prev alpha : Nat)
    (hRaw   : raw ≤ 1000)
    (hPrev  : prev ≤ 1000)
    (hAlpha : alpha ≤ 1000) :
    emaStep raw prev alpha ≤ 1000 := by
  unfold emaStep
  have h1 : alpha * raw ≤ alpha * 1000 := Nat.mul_le_mul_left alpha hRaw
  have h2 : (1000 - alpha) * prev ≤ (1000 - alpha) * 1000 := Nat.mul_le_mul_left (1000 - alpha) hPrev
  have h3 : alpha * 1000 + (1000 - alpha) * 1000 = 1000000 := by
    rw [← Nat.add_mul]
    have ha : alpha + (1000 - alpha) = 1000 := by omega
    rw [ha]
  omega

/-- Теорема 8b (EMA Convex Hull Invariant) -/
theorem emaStep_convex
    (raw prev alpha : Nat)
    (hAlpha : alpha ≤ 1000) :
    Nat.min raw prev ≤ emaStep raw prev alpha ∧
    emaStep raw prev alpha ≤ Nat.max raw prev := by
  unfold emaStep
  have hMinL : Nat.min raw prev ≤ raw := Nat.min_le_left raw prev
  have hMinR : Nat.min raw prev ≤ prev := Nat.min_le_right raw prev
  have hMaxL : raw ≤ Nat.max raw prev := Nat.le_max_left raw prev
  have hMaxR : prev ≤ Nat.max raw prev := Nat.le_max_right raw prev
  have hLow1 : alpha * Nat.min raw prev ≤ alpha * raw :=
    Nat.mul_le_mul_left alpha hMinL
  have hLow2 : (1000 - alpha) * Nat.min raw prev ≤ (1000 - alpha) * prev :=
    Nat.mul_le_mul_left (1000 - alpha) hMinR
  have hLowSum : alpha * Nat.min raw prev + (1000 - alpha) * Nat.min raw prev =
      1000 * Nat.min raw prev := by
    rw [← Nat.add_mul]
    have ha : alpha + (1000 - alpha) = 1000 := by omega
    rw [ha]
  have hHigh1 : alpha * raw ≤ alpha * Nat.max raw prev :=
    Nat.mul_le_mul_left alpha hMaxL
  have hHigh2 : (1000 - alpha) * prev ≤ (1000 - alpha) * Nat.max raw prev :=
    Nat.mul_le_mul_left (1000 - alpha) hMaxR
  have hHighSum : alpha * Nat.max raw prev + (1000 - alpha) * Nat.max raw prev =
      1000 * Nat.max raw prev := by
    rw [← Nat.add_mul]
    have ha : alpha + (1000 - alpha) = 1000 := by omega
    rw [ha]
  omega

private theorem runStreak_le_add_length (acc : Nat) (frames : List Bool) :
    runStreak acc frames ≤ acc + frames.length := by
  induction frames generalizing acc with
  | nil =>
    simp [runStreak]
  | cons b bs ih =>
    cases b with
    | false =>
      simp only [runStreak, List.foldl_cons, updateStreak, Bool.false_eq_true, ↓reduceIte, List.length_cons]
      have hRec := ih 0
      unfold runStreak at hRec
      omega
    | true =>
      simp only [runStreak, List.foldl_cons, updateStreak, ↓reduceIte, List.length_cons]
      have hRec := ih (acc + 1)
      unfold runStreak at hRec
      omega

private theorem runStreak_replicate_true (acc n : Nat) :
    runStreak acc (List.replicate n true) = acc + n := by
  induction n generalizing acc with
  | zero =>
    simp [runStreak]
  | succ n ih =>
    simp only [List.replicate_succ, runStreak, List.foldl_cons, updateStreak, ↓reduceIte]
    have hRec := ih (acc + 1)
    unfold runStreak at hRec
    omega

/-- Теорема 9a (Single-Frame Transient Spike Rejection) -/
theorem singleSpike_not_confirmed
    (confirmFrames : Nat)
    (hConfirm : 1 < confirmFrames) :
    isDangerConfirmed (runStreak 0 [true]) confirmFrames = false ∧
    isDangerConfirmed (runStreak 0 [true, false]) confirmFrames = false := by
  have hNot1 : ¬(1 ≥ confirmFrames) := by omega
  have hNot0 : ¬(0 ≥ confirmFrames) := by omega
  constructor
  · simp [runStreak, updateStreak, isDangerConfirmed, hNot1]
  · simp [runStreak, updateStreak, isDangerConfirmed, hNot0]

/-- Теорема 9b (Sub-Window Burst Rejection) -/
theorem shortBurst_not_confirmed
    (frames : List Bool) (confirmFrames : Nat)
    (hShort : frames.length < confirmFrames) :
    isDangerConfirmed (runStreak 0 frames) confirmFrames = false := by
  have hBound := runStreak_le_add_length 0 frames
  have hNot : ¬(runStreak 0 frames ≥ confirmFrames) := by omega
  unfold isDangerConfirmed
  simp [hNot]

/-- Теорема 9c (Deterministic Multi-Frame Hazard Confirmation) -/
theorem consecutiveHazard_confirms
    (initialStreak confirmFrames : Nat)
    (hPos : 1 ≤ confirmFrames) :
    isDangerConfirmed (runStreak initialStreak (List.replicate confirmFrames true)) confirmFrames = true := by
  rw [runStreak_replicate_true]
  have hGe : initialStreak + confirmFrames ≥ confirmFrames := by omega
  unfold isDangerConfirmed
  simp [hPos, hGe]

/-- Теорема 9d (Asymmetric Zone Classifier Hysteresis) -/
theorem classifyZone_hysteresis
    (score cautionThresh dangerThresh confirmFrames : Nat)
    (hConfirm : 1 < confirmFrames)
    (hHigh    : score ≥ dangerThresh) :
    (classifyZone score cautionThresh dangerThresh 0 confirmFrames).1 = .caution ∧
    (classifyZone score cautionThresh dangerThresh (confirmFrames - 1) confirmFrames).1 = .danger := by
  have hNot1 : ¬(0 + 1 ≥ confirmFrames) := by omega
  have hFull : (confirmFrames - 1) + 1 ≥ confirmFrames := by omega
  constructor
  · unfold classifyZone
    simp [hHigh, hNot1]
  · unfold classifyZone
    simp [hHigh, hFull]

end GlidanceSafety
