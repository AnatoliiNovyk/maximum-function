import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Параметри та початкові умови ---
alpha = 0.5
beta = 0.3
N = 1000000
S0 = 990000
I0 = 7000
R0 = 3000
t_start, t_end = 0, 25
t_span = (t_start, t_end) # Інтервал часу
t_eval = np.linspace(t_start, t_end, 500) # Точки часу для оцінки розв'язку

# Перевірка початкових умов
assert S0 + I0 + R0 == N, "Сума початкових умов не дорівнює N"

# --- Система диференціальних рівнянь SIR ---
def sir_model(t, y, alpha, beta):
  """
  Визначає систему диференціальних рівнянь SIR.
  y[0] -> S(t)
  y[1] -> I(t)
  y[2] -> R(t)
  """
  S, I, R = y
  dSdt = -alpha * S
  dIdt = alpha * S - beta * I
  dRdt = beta * I
  # Альтернативно, можна було б розв'язувати тільки для S та I,
  # а потім обчислити R = N - S - I. Розв'яжемо повну систему.
  return [dSdt, dIdt, dRdt]

# --- Розв'язання системи ODE ---
sol = solve_ivp(
    sir_model,
    t_span,
    [S0, I0, R0],
    args=(alpha, beta),
    dense_output=True, # Дозволяє отримати гладку функцію розв'язку
    t_eval=t_eval # Оцінити розв'язок у цих точках
)

# Отримання результатів
t = sol.t
S_num = sol.y[0]
I_num = sol.y[1]
R_num = sol.y[2] # R(t) = N - S(t) - I(t) також можна було б використати

# --- Побудова графіків чисельних розв'язків ---

# 1. Графік S(t)
plt.figure(figsize=(10, 6))
plt.plot(t, S_num, label='S(t) - Сприйнятливі (чисельний розв\'язок)')
plt.xlabel("Час (дні)")
plt.ylabel("Кількість індивідів")
plt.title("Динаміка сприйнятливих індивідів S(t)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Графік I(t)
plt.figure(figsize=(10, 6))
plt.plot(t, I_num, label='I(t) - Інфіковані (чисельний розв\'язок)', color='red')
plt.xlabel("Час (дні)")
plt.ylabel("Кількість індивідів")
plt.title("Динаміка інфікованих індивідів I(t)")
plt.legend()
plt.grid(True)
plt.show()

# 3. Графік R(t) (обчислено як N - S - I для перевірки, або взято з розв'язку)
R_calc = N - S_num - I_num
plt.figure(figsize=(10, 6))
# plt.plot(t, R_num, label='R(t) - Одужалі (чисельний розв\'язок)', color='green')
plt.plot(t, R_calc, label='R(t) - Одужалі (N-S-I)', color='green', linestyle='--')
plt.xlabel("Час (дні)")
plt.ylabel("Кількість індивідів")
plt.title("Динаміка одужалих індивідів R(t)")
plt.legend()
plt.grid(True)
plt.show()


# 4. Всі три графіки разом (чисельні розв'язки)
plt.figure(figsize=(12, 7))
plt.plot(t, S_num, label='S(t) - Сприйнятливі')
plt.plot(t, I_num, label='I(t) - Інфіковані', color='red')
plt.plot(t, R_num, label='R(t) - Одужалі', color='green')
plt.xlabel("Час (дні)")
plt.ylabel("Кількість індивідів")
plt.title("SIR Модель Епідемії (Чисельний Розв'язок)")
plt.legend()
plt.grid(True)
plt.show()

# --- Апроксимація методом найменших квадратів (МНК) ---

# Функція для S(t)
def s_fit_func(t, s0_fit, alpha_fit):
  """Функціональна форма для апроксимації S(t)."""
  return s0_fit * np.exp(-alpha_fit * t)

# Функція для I(t) - використовуємо форму з завдання
# Зверніть увагу: ця форма є точною лише при alpha = beta
def i_fit_func(t, i0_fit, alpha_fit, s0_fit):
   """Функціональна форма для апроксимації I(t) згідно завдання."""
   # Використовуємо alpha_fit з параметрів функції, а не глобальний alpha
   return (i0_fit + alpha_fit * s0_fit * t) * np.exp(-alpha_fit * t)

# Початкові припущення для параметрів МНК
# Для S(t): використовуємо відомі S0 та alpha
initial_guess_s = [S0, alpha]
# Для I(t): використовуємо відомі I0, alpha, S0
initial_guess_i = [I0, alpha, S0]


try:
    # МНК для S(t)
    params_s, covariance_s = curve_fit(s_fit_func, t, S_num, p0=initial_guess_s)
    s0_fit, alpha_fit_s = params_s
    S_fit = s_fit_func(t, s0_fit, alpha_fit_s)
    print("\n--- Результати МНК для S(t) ---")
    print(f"  Підігнані параметри: S0 = {s0_fit:.2f}, alpha = {alpha_fit_s:.4f}")
    print(f"  Отримана функція: S(t) = {s0_fit:.2f} * exp(-{alpha_fit_s:.4f} * t)")

    # МНК для I(t)
    # Обережно: форма може погано підходити, якщо alpha != beta
    params_i, covariance_i = curve_fit(i_fit_func, t, I_num, p0=initial_guess_i, maxfev=5000) # Збільшуємо maxfev
    i0_fit, alpha_fit_i, s0_fit_i = params_i
    I_fit = i_fit_func(t, i0_fit, alpha_fit_i, s0_fit_i)
    print("\n--- Результати МНК для I(t) ---")
    print(f"  Підігнані параметри: I0 = {i0_fit:.2f}, alpha = {alpha_fit_i:.4f}, S0 = {s0_fit_i:.2f}")
    print(f"  Отримана функція: I(t) = ({i0_fit:.2f} + {alpha_fit_i:.4f} * {s0_fit_i:.2f} * t) * exp(-{alpha_fit_i:.4f} * t)")


    # R(t) з МНК
    R_fit = N - S_fit - I_fit
    # Переконаємось, що R_fit не стає від'ємним через неточності апроксимації
    R_fit = np.maximum(R_fit, 0)

    print("\n--- Рівняння для R(t) (з МНК) ---")
    print(f"  R(t) = N - S_fit(t) - I_fit(t)")


    # --- Побудова графіків МНК ---
    plt.figure(figsize=(12, 7))
    plt.plot(t, S_num, '--', label='S(t) (чисельний)', alpha=0.5)
    plt.plot(t, I_num, '--', label='I(t) (чисельний)', color='red', alpha=0.5)
    plt.plot(t, R_num, '--', label='R(t) (чисельний)', color='green', alpha=0.5)

    plt.plot(t, S_fit, label=f'S(t) (МНК)', color='blue')
    plt.plot(t, I_fit, label=f'I(t) (МНК)', color='orange')
    plt.plot(t, R_fit, label=f'R(t) (МНК)', color='purple')

    plt.xlabel("Час (дні)")
    plt.ylabel("Кількість індивідів")
    plt.title("SIR Модель: Чисельний розв'язок vs Апроксимація МНК")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Починаємо вісь Y з 0
    plt.show()

except RuntimeError as e:
    print(f"\nПомилка під час МНК: {e}")
    print("Не вдалося знайти оптимальні параметри для апроксимації.")
    # Якщо МНК не вдалося, ми не можемо продовжити з цими частинами
    S_fit, I_fit, R_fit = None, None, None


# --- Знаходження часу максимальної інфекції (t_max) ---
# Потрібно знайти максимум функції I(t) (чисельного розв'язку)

# Створюємо інтерполяційну функцію для I_num, щоб minimize_scalar міг її використовувати
I_interp = interp1d(t, I_num)

# minimize_scalar шукає мінімум, тому шукаємо мінімум від -I(t)
# Обмежуємо пошук інтервалом часу [t_start, t_end]
result_min = minimize_scalar(lambda t_val: -I_interp(t_val), bounds=(t_start, t_end), method='bounded')

if result_min.success:
    t_max = result_min.x
    I_max = I_interp(t_max) # або -result_min.fun
    print("\n--- Максимальна кількість інфікованих ---")
    print(f"  Час досягнення максимуму (t_max): {t_max:.4f} днів")
    print(f"  Максимальна кількість інфікованих (I_max): {I_max:.2f}")

    # Позначимо максимум на графіку I(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t, I_num, label='I(t) - Інфіковані (чисельний розв\'язок)', color='red')
    plt.scatter([t_max], [I_max], color='black', zorder=5, label=f'Максимум I(t) при t={t_max:.2f}')
    plt.xlabel("Час (дні)")
    plt.ylabel("Кількість індивідів")
    plt.title("Динаміка інфікованих індивідів I(t) з позначкою максимуму")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("\nНе вдалося знайти максимум функції I(t) за допомогою minimize_scalar.")
    print(f"  Повідомлення оптимізатора: {result_min.message}")
