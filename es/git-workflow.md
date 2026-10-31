# Flujo de Git y copia limpia

## Convenciones de desarrollo

Usa [Conventional Commits](https://www.conventionalcommits.org/) para los
mensajes, por ejemplo `docs: document model access` o
`test: cover tensor conversion`. Trabaja en una rama de tarea como
`codex/issue-4-clean-checkout` y abre un pull request contra la rama
predeterminada. Utiliza la plantilla del PR: qué, por qué, cómo, diagrama
Mermaid, cómo probarlo, archivos modificados y objetivos cumplidos. El inglés va
primero y la sección en español al final. La documentación en español va en
`es/`.

## Metadatos AppleDouble

La copia original en el volumen externo mostró errores de Git relacionados con
un archivo `._pack-*.idx` dentro de `.git/objects/pack`. Las entradas `._*`
son metadatos AppleDouble; una regla de exclusión afecta solo al árbol de
trabajo y no repara metadatos dentro de `.git`.

Usa una copia nueva en el disco local cuando el volumen externo introduzca estos
archivos. Elige una ubicación adecuada fuera de otra copia existente:

```sh
git clone https://github.com/LuigimonSoft/idol-sdk-MLX.git
cd idol-sdk-MLX
git status --short --branch
git fsck --no-reflogs --connectivity-only
```

La copia existente puede contener cambios sin guardar: revísalos y consérvalos
antes de trasladar trabajo. No borres ni modifiques metadatos `.git` de otra
copia como parte de este procedimiento. El `.gitignore` de la nueva copia
excluye AppleDouble del árbol de trabajo, entornos virtuales locales y cachés de
Python; no impide que el sistema de archivos cree entradas dentro de `.git`.

## Validación

Una copia separada en el sistema de archivos temporal local superó `git status`
y `git fsck --no-reflogs --connectivity-only`. Esto comprueba la conectividad
de los objetos Git, no todos los posibles fallos del disco. Una copia temporal
no es almacenamiento permanente; clona en un directorio local permanente para
trabajar de forma continuada.

Versión principal: [docs/git-workflow.md](../docs/git-workflow.md).
