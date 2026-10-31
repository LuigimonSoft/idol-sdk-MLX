# Inventario de acceso y licencias de modelos

Comprobación realizada el 30 de octubre de 2026 en el Mac de referencia, mediante
solicitudes de metadatos a Hugging Face sin autenticación. Es un inventario, no
una interpretación legal ni una garantía de acceso futuro. Revisa las
condiciones vigentes antes de usar o redistribuir los modelos.

| Componente | Fuente | Licencia publicada | Acceso observado |
| --- | --- | --- | --- |
| Pesos Idol | [speridlabs/idol](https://huggingface.co/speridlabs/idol) | Apache-2.0 según los metadatos; [licencia del SDK original](https://github.com/speridlabs/idol-sdk/blob/main/LICENSE). | Los metadatos de los cinco archivos necesarios son públicos. |
| DINOv3 ViT-H+ | [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | Licencia DINOv3 indicada en la ficha; revisa las condiciones actuales en la página del modelo. | Acceso restringido con aprobación manual. Una solicitud sin autenticación a `model.safetensors` devolvió HTTP 401. |
| VAE de FLUX.2 dev | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | Licencia FLUX no comercial indicada en la ficha; consulta la [licencia del repositorio](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md) y las condiciones de acceso. | Acceso restringido, con aprobación automática tras aceptar las condiciones. Una solicitud sin autenticación al archivo `vae/diffusion_pytorch_model.safetensors` devolvió HTTP 401. |

El SDK usa el VAE de FLUX.2 dev. No supongas que otro modelo FLUX, su VAE o un
checkpoint convertido tiene la misma licencia o es intercambiable. Revisa las
condiciones del modelo y del componente para el uso y la distribución previstos.
La ficha de FLUX.2 dev indica una licencia no comercial; verifica los derechos
comerciales y de producción directamente con el titular.

La licencia DINOv3 también incluye condiciones de redistribución y una restricción sobre ingeniería inversa. Revisa la licencia antes de implementar o distribuir una adaptación; este inventario no determina cómo se aplican esas disposiciones al proyecto.

## Archivos comprobados

El SDK original solicita `config.json`, `denoiser.safetensors`,
`decoder.safetensors`, `decoder_gs.safetensors` y `birefnet.safetensors`
del repositorio Idol. Los metadatos de los cinco fueron accesibles sin token. No
se descargaron los archivos completos ni se verificó su integridad.

No había un token de Hugging Face disponible en el entorno de comprobación. Los
HTTP 401 demuestran que el acceso sin autenticación está bloqueado, pero no
permiten afirmar si alguna cuenta concreta ya dispone de aprobación.

## Habilitar y verificar el acceso

1. Inicia sesión en Hugging Face y revisa las condiciones de las páginas de
   DINOv3 y FLUX.2 enlazadas arriba. Solicita o acepta el acceso correspondiente;
   DINOv3 puede requerir aprobación.
2. Autentícate localmente con `hf auth login`. Usa un token con acceso de
   lectura autorizado para repositorios restringidos. No publiques tokens en
   issues, registros, commits ni documentación.
3. Comprueba la cuenta con `hf auth whoami` y repite la comprobación desde el
   entorno de referencia después de obtener acceso. Un error de red o límite de
   solicitudes no demuestra una denegación.
4. Mantén los pesos en una caché local y respeta las condiciones de cada modelo.
   Este repositorio no los incluye.

Tras iniciar sesión, verifica el acceso sin descargar los pesos con el CLI
`hf` del entorno de referencia:

```sh
hf download facebook/dinov3-vith16plus-pretrain-lvd1689m model.safetensors --dry-run
hf download black-forest-labs/FLUX.2-dev vae/diffusion_pytorch_model.safetensors --dry-run
```

La descarga completa, la validación de integridad y la inferencia no forman
parte de este inventario. El issue #6 puede resolverse identificando el acceso
pendiente; conseguir la aprobación es un requisito de la validación posterior.

Versión principal: [docs/model-access.md](../docs/model-access.md).
