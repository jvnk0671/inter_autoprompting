import { useState, useEffect, useRef } from 'react';
import {
  Copy, RefreshCw, AlertCircle, ArrowDownCircle, Plus, MessageSquare, ChevronDown, Check
} from 'lucide-react';

const CustomSelect = ({ label, value, options, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative flex-1" ref={containerRef}>
      <label className="block text-[10px] font-bold text-gray-500 mb-1.5 uppercase tracking-widest">{label}</label>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3 bg-gray-900 text-gray-200 border border-gray-700 rounded-lg hover:border-gray-500 transition-all text-sm"
      >
        <span className="truncate">{options.find(opt => opt.value === value)?.label}</span>
        <ChevronDown size={16} className={`transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full mt-2 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-100">
          {options.map((opt) => (
            <div
              key={opt.value}
              onClick={() => {
                onChange(opt.value);
                setIsOpen(false);
              }}
              className="flex items-center justify-between px-4 py-3 text-sm cursor-pointer hover:bg-indigo-600 transition-colors"
            >
              {opt.label}
              {value === opt.value && <Check size={14} className="text-white" />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const RefinedPromptingService = () => {
  const [chats] = useState([
    { id: 1, title: 'Анализ тыквенного супа', date: 'Сегодня' },
    { id: 2, title: 'Оптимизация кода C++', date: 'Вчера' },
    { id: 3, title: 'Перевод JSON в Parquet', date: '15 Апр' }
  ]);
  const [activeChatId, setActiveChatId] = useState(1);

  const [llm, setLlm] = useState('gpt-4');
  const [method, setMethod] = useState('chain-of-thought');
  const [inputPrompt, setInputPrompt] = useState('');
  const [maxLength, setMaxLength] = useState(300);
  const [outputPrompt, setOutputPrompt] = useState('');
  const [isOptimizing, setIsOptimizing] = useState(false);

  const llmOptions = [
    { value: 'gpt-4', label: 'GPT-4 Turbo' },
    { value: 'claude-3', label: 'Claude 3.5 Sonnet' },
    { value: 'llama-3', label: 'Llama 3 (70B)' }
  ];

  const methodOptions = [
    { value: 'chain-of-thought', label: 'Chain of Thought' },
    { value: 'expert-role', label: 'Экспертная роль' },
    { value: 'few-shot', label: 'Few-Shot примеры' }
  ];

  const handleOptimize = () => {
    if (!inputPrompt) return;
    setIsOptimizing(true);
    setTimeout(() => {
      setOutputPrompt(`[ОПТИМИЗИРОВАНО]: Ты — эксперт. Твоя задача: ${inputPrompt}. Используй пошаговое рассуждение.`);
      setIsOptimizing(false);
    }, 1200);
  };

  return (
    <div className="flex h-screen bg-gray-950 font-sans text-gray-200 overflow-hidden">

      <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col hidden md:flex">
        <div className="p-4"><button className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-750 transition-colors text-xs font-bold"><Plus size={14} /> NEW PROMPT</button></div>
        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          {chats.map(chat => (
            <div key={chat.id} onClick={() => setActiveChatId(chat.id)} className={`flex items-center gap-3 px-3 py-3 rounded-lg cursor-pointer transition-all ${activeChatId === chat.id ? 'bg-indigo-950/40 text-indigo-400' : 'text-gray-500 hover:bg-gray-800'}`}>
              <MessageSquare size={16} /><span className="truncate text-sm">{chat.title}</span>
            </div>
          ))}
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="max-w-2xl mx-auto py-12 px-6 space-y-8">

          <div className="flex gap-4">
            <CustomSelect label="модель" value={llm} options={llmOptions} onChange={setLlm} />
            <CustomSelect label="метод" value={method} options={methodOptions} onChange={setMethod} />
          </div>

          <div className="bg-gray-800 p-1 rounded-xl border border-gray-700 shadow-2xl focus-within:border-indigo-500 transition-all">
            <textarea
              className="w-full h-40 p-4 bg-transparent text-gray-200 outline-none resize-none text-sm leading-relaxed"
              placeholder="Ваш предварительный промпт."
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
            />
          </div>

          <div className="flex justify-center -my-10 relative z-20">
            <button
              onClick={handleOptimize}
              disabled={isOptimizing || !inputPrompt}
              className={`w-16 h-16 rounded-full flex items-center justify-center transition-all border-[6px] shadow-2xl ${isOptimizing ? 'bg-gray-700 border-indigo-900 animate-pulse' : 'bg-gray-800 text-indigo-400 border-gray-950 hover:scale-110 active:scale-90'
                }`}
            >
              {isOptimizing ? <RefreshCw className="animate-spin" size={24} /> : <ArrowDownCircle size={32} />}
            </button>
          </div>

          <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 space-y-4">
            <div className="flex justify-between items-center"><span className="text-[10px] font-bold text-gray-500 uppercase">Максимальная длина</span><span className="text-indigo-400 font-mono text-xs">{maxLength} ch.</span></div>
            <input type="range" min="50" max="500" value={maxLength} onChange={(e) => setMaxLength(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none accent-indigo-500 cursor-pointer" />
          </div>

          {outputPrompt ? (
            <div className="animate-in slide-in-from-bottom-4 duration-500">
              <div className="bg-indigo-950/20 border border-indigo-500/30 rounded-xl p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Результат</span>
                  <button onClick={() => navigator.clipboard.writeText(outputPrompt)} className="text-[10px] flex items-center gap-2 text-gray-400 hover:text-white transition"><Copy size={12} /> COPY</button>
                </div>
                <textarea readOnly value={outputPrompt} className="w-full bg-transparent text-sm text-gray-300 font-mono leading-relaxed outline-none h-32 resize-none" />
              </div>
            </div>
          ) : (
            <div className="text-center py-10 opacity-20 flex flex-col items-center gap-3"><AlertCircle size={40} /><span className="text-xs uppercase tracking-tighter">Ожидание ввода...</span></div>
          )}
        </div>
      </main>
    </div>
  );
};

export default RefinedPromptingService;