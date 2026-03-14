<div align="center">

# Miceagent Extension
<p align="center">
  <img src="Asset/miceagent-icon.png" alt="MiceAgent" width="180" />
</p>


**Agente inteligente dentro de uma extensão web**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange?style=for-the-badge)](#)

Backend para uma extensão web com foco em automação, orquestração de ações e integração com modelos de IA.

</div>

---

## Sobre o projeto

O **Miceagent Extension** é um backend em Python criado para dar suporte a uma extensão web inteligente.

A proposta do projeto é servir como núcleo de processamento da extensão, conectando interface, lógica de agente, organização modular e futuras integrações com ferramentas, automações e provedores de IA.

---

## Destaques

- Estrutura em Python
- Organização modular em `src/`
- Testes automatizados em `tests/`
- Backend preparado para evolução contínua
- Projeto voltado para agente inteligente em extensão web
- Estrutura simples para desenvolvimento local e expansão futura

---

## Estrutura do repositório

```bash
Miceagent-Extension/
├── src/
├── tests/
├── .gitignore
├── .python-version
├── README.md
├── environment.yml
├── main.py
├── pyproject.toml
├── pyrightconfig.json
├── server.py
└── uv.lock
```
---
## Organização do código
A base do projeto está separada de forma a facilitar manutenção, testes e crescimento da aplicação.

Diretórios principais:
- src/ → código-fonte principal

- tests/ → testes automatizados

- main.py → ponto de entrada principal

- server.py → execução do servidor

- pyproject.toml → configuração do projeto Python

- environment.yml → definição de ambiente

- pyrightconfig.json → configuração de tipagem estática
---
## Instalação
1. Clone o repositório
```bash
git clone https://github.com/AndreRachid-rgb/Miceagent-Extension.git
cd Miceagent-Extension
```
2. Configure o ambiente
Se estiver usando uv:
```bash
uv sync
```
Se preferir ambiente virtual tradicional:
```bash
python -m venv .venv
```
3. Ative o ambiente
No Windows:
```bash
.venv\Scripts\activate
```
No Linux/macOS:
```bash
source .venv/bin/activate
```
4. Instale as dependências (TOML)
```bash
pip install .
```
5. Executar
Você pode iniciar o projeto pelo arquivo:
```bash
uv run server.py
```
---
<div align="center">
   <img src="https://github.com/AndreRachid-rgb.png" width="100px" style="border-radius: 50%;" alt="André Rachid"/>

  André Rachid - Desenvolvedor <br>
  [![GitHub](https://img.shields.io/badge/GitHub-Perfil-181717?style=flat&logo=github)](https://github.com/AndreRachid-rgb)
</div>
