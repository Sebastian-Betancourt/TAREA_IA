import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad
np.random.seed(42)

n = 200
horas = np.random.uniform(0, 10, size=n)                   # 0–10 horas
nivel_prev = np.random.randint(1, 11, size=n)              # 1–10
asistencia = np.random.uniform(50, 100, size=n)            # 50–100%
prom_tareas = np.random.uniform(0, 10, size=n)             # 0–10
tipo_est = np.random.choice([0, 1], size=n)                # 0=grupo, 1=solo

# Combinación lineal + ruido para determinar probabilidad de aprobar
logit = (
    0.3*horas +
    0.2*nivel_prev +
    0.2*(asistencia/10) +
    0.2*prom_tareas +
    0.1*tipo_est
)
prob = 1 / (1 + np.exp(- (logit - 5)))  # Sigmoide centrada
etiqueta = (prob > 0.5).astype(int)     # 1 = Aprobado, 0 = No aprobado

df = pd.DataFrame({
    'HorasEstudio': horas,
    'NivelPrevio': nivel_prev,
    'Asistencia': asistencia,
    'PromTareas': prom_tareas,
    'TipoEstudiante': tipo_est,
    'Resultado': etiqueta
})

df.to_csv('dataset_estudiantes.csv', index=False)
print(df.head())


