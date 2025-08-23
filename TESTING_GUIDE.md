# Testing Guide for Financial Q&A System

## 🧪 **Quick Testing Commands**

### **1. Test RAG System**

```bash
# Test RAG functionality
python -c "
import sys
sys.path.append('.')
from src.rag_system.integrated_rag import IntegratedRAGSystem

print('Testing RAG System...')
rag = IntegratedRAGSystem()
result = rag.process_query('What was Apple revenue in 2023?')
print(f'Answer: {result[\"answer\"]}')
print(f'Confidence: {result[\"confidence\"]}')
print(f'Response Time: {result[\"response_time\"]:.3f}s')
"
```

### **2. Test MoE Fine-Tuned Model**

```bash
# Test MoE system
python -c "
import sys
sys.path.append('.')
from src.fine_tuning.fine_tuner import FineTuner

print('Testing MoE System...')
ft_model = FineTuner(use_moe=True)
print(f'MoE Available: {ft_model.moe_system is not None}')
print(f'MoE Trained: {ft_model.moe_system.is_trained if ft_model.moe_system else False}')

if ft_model.moe_system and ft_model.moe_system.is_trained:
    result = ft_model.process_query('What was the revenue growth?')
    print(f'Answer: {result[\"answer\"]}')
    print(f'Confidence: {result[\"confidence\"]}')
"
```

### **3. Test Both Systems Comparison**

```bash
# Compare RAG vs MoE
python -c "
import sys
sys.path.append('.')
from src.rag_system.integrated_rag import IntegratedRAGSystem
from src.fine_tuning.fine_tuner import FineTuner

query = 'What was Apple total revenue in 2023?'
print(f'Query: {query}')
print('=' * 50)

# Test RAG
print('RAG System:')
rag = IntegratedRAGSystem()
rag_result = rag.process_query(query)
print(f'  Answer: {rag_result[\"answer\"]}')
print(f'  Confidence: {rag_result[\"confidence\"]}')
print(f'  Time: {rag_result[\"response_time\"]:.3f}s')

print()

# Test MoE
print('MoE System:')
ft_model = FineTuner(use_moe=True)
if ft_model.moe_system and ft_model.moe_system.is_trained:
    moe_result = ft_model.process_query(query)
    print(f'  Answer: {moe_result[\"answer\"]}')
    print(f'  Confidence: {moe_result[\"confidence\"]}')
else:
    print('  MoE system not available or not trained')
"
```

## 🔍 **System Status Checks**

### **Check MoE Models**

```bash
# Check if MoE models exist
echo "MoE Models Status:"
if [ -d "models/fine_tuned/moe" ]; then
    echo "✅ MoE directory exists"
    ls -la models/fine_tuned/moe/
    if [ -f "models/fine_tuned/moe/moe_config.json" ]; then
        echo "✅ MoE config found:"
        cat models/fine_tuned/moe/moe_config.json
    else
        echo "❌ MoE config missing"
    fi
else
    echo "❌ MoE directory not found"
fi
```

### **Check RAG Data**

```bash
# Check RAG data availability
echo "RAG Data Status:"
echo "Processed files:"
ls -la data/processed/ 2>/dev/null || echo "❌ No processed data"
echo ""
echo "QA pairs:"
ls -la data/qa_pairs/ 2>/dev/null || echo "❌ No QA pairs"
```

### **Check Dependencies**

```bash
# Check Python dependencies
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    print('❌ PyTorch not available')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not available')

try:
    import langchain
    print(f'✅ LangChain: {langchain.__version__}')
except ImportError:
    print('❌ LangChain not available')

try:
    import streamlit
    print(f'✅ Streamlit: {streamlit.__version__}')
except ImportError:
    print('❌ Streamlit not available')
"
```

## 🚀 **Run Full Test Suite**

### **All Tests**

```bash
# Run all tests
python scripts/test_runner.py
```

### **Specific Test Categories**

```bash
# RAG tests only
python -m pytest tests/rag_system/ -v

# Fine-tuning tests only
python -m pytest tests/fine_tuning/ -v

# Integration tests
python -m pytest tests/test_integration.py -v
```

## 🎯 **Performance Testing**

### **Evaluation on Test Questions**

```bash
# Run evaluation
python -c "
import sys
sys.path.append('.')
from src.evaluation.evaluator import Evaluator

print('Running evaluation...')
evaluator = Evaluator()
results = evaluator.evaluate_both_systems()
print(f'Results: {results}')
"
```

### **Speed Benchmarks**

```bash
# Benchmark response times
python -c "
import sys
import time
sys.path.append('.')
from src.rag_system.integrated_rag import IntegratedRAGSystem

queries = [
    'What was Apple revenue?',
    'What were the main expenses?',
    'How did the company perform?'
]

rag = IntegratedRAGSystem()
total_time = 0

for query in queries:
    start = time.time()
    result = rag.process_query(query)
    elapsed = time.time() - start
    total_time += elapsed
    print(f'{query}: {elapsed:.3f}s')

print(f'Average: {total_time/len(queries):.3f}s')
"
```

## 🐛 **Debugging Commands**

### **Check Logs**

```bash
# Check for errors in recent runs
python -c "
import logging
logging.basicConfig(level=logging.INFO)

# Test with verbose logging
import sys
sys.path.append('.')
from src.rag_system.integrated_rag import IntegratedRAGSystem

rag = IntegratedRAGSystem()
result = rag.process_query('test query')
"
```

### **Memory Usage**

```bash
# Check memory usage
python -c "
import psutil
import os

process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')

# Load systems and check memory
import sys
sys.path.append('.')
from src.rag_system.integrated_rag import IntegratedRAGSystem
from src.fine_tuning.fine_tuner import FineTuner

rag = IntegratedRAGSystem()
print(f'After RAG: {process.memory_info().rss / 1024 / 1024:.1f} MB')

ft_model = FineTuner(use_moe=True)
print(f'After MoE: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## 📊 **Sample Test Queries**

### **Financial Questions**

```bash
# Test various financial queries
python -c "
import sys
sys.path.append('.')
from src.rag_system.integrated_rag import IntegratedRAGSystem

queries = [
    'What was Apple total revenue in 2023?',
    'What were the main operating expenses?',
    'How much did the company spend on R&D?',
    'What was the net income for fiscal 2023?',
    'What were the key financial highlights?'
]

rag = IntegratedRAGSystem()
for query in queries:
    result = rag.process_query(query)
    print(f'Q: {query}')
    print(f'A: {result[\"answer\"]}')
    print(f'Confidence: {result[\"confidence\"]}')
    print('-' * 50)
"
```

## 🔧 **Setup Testing**

### **Test Setup Wizard Components**

```bash
# Test setup wizard methods
python -c "
import sys
sys.path.append('.')
from src.ui.setup_wizard import SetupWizard

# Test if methods exist
methods = ['_run_real_rag_initialization', '_run_real_moe_training']
for method in methods:
    if hasattr(SetupWizard, method):
        print(f'✅ {method} exists')
    else:
        print(f'❌ {method} missing')
"
```

## 🎉 **Quick Success Test**

### **One-Command Full Test**

```bash
# Complete system test
python -c "
import sys
sys.path.append('.')

print('🧪 COMPLETE SYSTEM TEST')
print('=' * 50)

# Test 1: RAG System
try:
    from src.rag_system.integrated_rag import IntegratedRAGSystem
    rag = IntegratedRAGSystem()
    result = rag.process_query('What was Apple revenue?')
    print('✅ RAG System: Working')
    print(f'   Answer: {result[\"answer\"][:100]}...')
except Exception as e:
    print(f'❌ RAG System: Failed - {e}')

# Test 2: MoE System
try:
    from src.fine_tuning.fine_tuner import FineTuner
    ft_model = FineTuner(use_moe=True)
    if ft_model.moe_system and ft_model.moe_system.is_trained:
        result = ft_model.process_query('What was the revenue?')
        print('✅ MoE System: Working')
        print(f'   Answer: {result[\"answer\"][:100]}...')
    else:
        print('⚠️  MoE System: Available but not trained')
except Exception as e:
    print(f'❌ MoE System: Failed - {e}')

print('\\n🎯 Test Complete!')
"
```

---

## 📝 **Usage Notes**

- Run tests from the project root directory
- Ensure virtual environment is activated
- Some tests require trained models to be present
- Use `-v` flag with pytest for verbose output
- Check `requirements.txt` if import errors occur

**Happy Testing! 🚀**
