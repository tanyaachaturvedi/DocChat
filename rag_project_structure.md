rag_project/
├── app.py
├── requirements.txt
├── config/
│   ├── settings.py
│   └── prompts.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── text_processor.py
│   │   └── vector_store.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── bedrock_client.py
│   │   └── rag_chain.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── data/
│   ├── raw/
│   └── processed/
├── vector_store/
├── logs/
├── docs/
│   ├── README.md
│   └── setup.md
└── .env
