import { useState, useEffect, useRef } from 'react';
import {
  Copy, RefreshCw, AlertCircle, ArrowDownCircle, Plus, MessageSquare, ChevronDown, Check, Send
} from 'lucide-react';

const CustomSelect = ({ label, value, options, onChange, hint }) => {
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
      <div className="flex justify-between items-end mb-1.5 px-1">
        <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-widest">{label}</label>
        {hint && <span className="text-[9px] text-indigo-400 italic">{hint}</span>}
      </div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-2.5 bg-gray-900 text-gray-200 border border-gray-700 rounded-lg hover:border-gray-500 transition-all text-xs"
      >
        <span className="truncate">{options.find(opt => opt.value === value)?.label}</span>
        <ChevronDown size={14} className={`transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
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
              className="flex items-center justify-between px-3 py-2.5 text-xs cursor-pointer hover:bg-indigo-600 transition-colors"
            >
              {opt.label}
              {value === opt.value && <Check size={12} className="text-white" />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const RefinedPromptingService = () => {
  const [chats] = useState([
    { id: 1, title: 'Chat 1', date: 'today' },
    { id: 2, title: 'Chat 2', date: 'erm ago' },
    { id: 3, title: 'Chat 3', date: '1984' }
  ]);
  const [activeChatId, setActiveChatId] = useState(1);

  const [llm, setLlm] = useState('gpt-4');
  const [method, setMethod] = useState('example');
  const [coolMode, setCoolMode] = useState('hype');
  const [targetModel, setTargetModel] = useState('meta-llama/llama-3.3-70b-instruct');
  const [systemModel, setSystemModel] = useState('nvidia/nemotron-3-nano-omni-30b-a3b-reasoning');
  const [inputPrompt, setInputPrompt] = useState('');
  const [maxLength, setMaxLength] = useState(300);
  const [outputPrompt, setOutputPrompt] = useState('');
  const [isOptimizing, setIsOptimizing] = useState(false);

  const llmOptions = [{ value: 'gpt-4', label: 'GPT-4 Turbo' }, { value: 'claude-3', label: 'Claude 3.5' }];
  const methodOptions = [{ value: 'example', label: 'Example' }, { value: 'coolprompt', label: 'CoolPrompt' }, { value: 'promptomatix', label: 'Promptomatix' }];
  const coolModeOptions = [{ value: 'hype', label: 'Hype' }, { value: 'distill', label: 'Distill' }];
  const advancedModelOptions = [
    { value: 'meta-llama/llama-3.3-70b-instruct', label: 'Llama 3.3 70B' },
    { value: 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', label: 'Nemotron 3 Nano' }
  ];

  const handleOptimize = async () => {
    if (!inputPrompt.trim()) return;
    setIsOptimizing(true);
    setOutputPrompt('');

    try {
      const response = await fetch('http://localhost:8000/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: inputPrompt,
          method,
          ch_limit: Number(maxLength),
          uncertainty: 20,
          target_model: targetModel,
          system_model: systemModel,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Backend error');
      }

      setOutputPrompt(data.optimized_prompt);
    } catch (error) {
      setOutputPrompt(`Ошибка: ${error.message}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-950 font-sans text-gray-200 overflow-hidden">
      
      <aside className="w-60 bg-gray-950 border-r border-gray-800 flex flex-col flex-shrink-0">
        <div className="p-4"><button className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-gray-900 border border-gray-700 rounded-lg hover:bg-gray-800 transition-colors text-[10px] font-bold"><Plus size={14} /> NEW CHAT</button></div>
        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          {chats.map(chat => (
            <div key={chat.id} onClick={() => setActiveChatId(chat.id)} className={`flex items-center gap-3 px-3 py-3 rounded-lg cursor-pointer transition-all ${activeChatId === chat.id ? 'bg-indigo-950/40 text-indigo-400' : 'text-gray-500 hover:bg-gray-900'}`}>
              <MessageSquare size={14} /><span className="truncate text-xs">{chat.title}</span>
            </div>
          ))}
        </div>
      </aside>

      <section className="flex-1 flex flex-col bg-gray-900/20 relative">
        <header className="h-14 border-b border-gray-800 flex items-center px-6">
          <h2 className="text-sm font-semibold text-gray-400">Чат: {chats.find(c => c.id === activeChatId)?.title}</h2>
        </header>
        
        <div className="flex-1 overflow-y-auto p-6 space-y-4 text-sm">
          <div className="max-w-2xl mx-auto opacity-30 text-center mt-20 italic">Здесь будет история вашего диалога...</div>
        </div>

        <div className="p-6 border-t border-gray-800 bg-gray-950/50">
          <div className="max-w-3xl mx-auto relative">
            <input 
              type="text" 
              placeholder="Спросить о..." 
              className="w-full bg-gray-900 border border-gray-700 rounded-xl py-3 px-4 pr-12 outline-none focus:border-indigo-500 transition-all text-sm"
            />
            <button className="absolute right-3 top-2.5 p-1 text-indigo-500 hover:text-indigo-400">
              <Send size={20} />
            </button>
          </div>
        </div>
      </section>

      <aside className="w-[450px] bg-gray-950 border-l border-gray-800 overflow-y-auto p-6 flex-shrink-0">
        <div className="space-y-6">
          <div className="flex items-center gap-2 mb-4">
            <h3 className="text-xs font-black uppercase tracking-widest text-white">Автопромптинг</h3>
          </div>

          <div className="space-y-4">
            <div className="flex gap-3">
              <CustomSelect label="модель" value={llm} options={llmOptions} onChange={setLlm} />
              <CustomSelect label="метод" value={method} options={methodOptions} onChange={setMethod} />
            </div>

            {method === 'coolprompt' && (
              <div className="p-4 bg-gray-900/30 rounded-xl border border-gray-800/50 space-y-4 animate-in fade-in slide-in-from-top-2">
                <CustomSelect label="Режим CoolPrompt" value={coolMode} options={coolModeOptions} onChange={setCoolMode} />
                <div className="flex gap-3">
                  <CustomSelect label="Target Model" value={targetModel} options={advancedModelOptions} onChange={setTargetModel} />
                  <CustomSelect label="System Model" value={systemModel} options={advancedModelOptions} onChange={setSystemModel} hint="(рек: llama)" />
                </div>
              </div>
            )}
          </div>

          <div className="bg-gray-900 rounded-xl border border-gray-700 focus-within:border-indigo-500 transition-all overflow-hidden shadow-2xl">
            <textarea
              className="w-full h-32 p-4 bg-transparent text-gray-200 outline-none resize-none text-sm leading-relaxed"
              placeholder="Ваш предварительный промпт."
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
            />
          </div>

          <div className="flex justify-center relative py-2">
            <button
              onClick={handleOptimize}
              disabled={isOptimizing || !inputPrompt}
              className={`w-14 h-14 rounded-full flex items-center justify-center transition-all border-[4px] shadow-xl ${isOptimizing ? 'bg-gray-700 border-indigo-900 animate-pulse' : 'bg-gray-800 text-indigo-400 border-gray-950 hover:scale-105 active:scale-95'}`}
            >
              {isOptimizing ? <RefreshCw className="animate-spin" size={20} /> : <ArrowDownCircle size={28} />}
            </button>
          </div>

          <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-800 space-y-3">
            <div className="flex justify-between items-center text-[10px] font-bold text-gray-500 uppercase">
              <span>Макс. длина</span>
              <span className="text-indigo-400 font-mono">{maxLength} ch.</span>
            </div>
            <input type="range" min="50" max="500" value={maxLength} onChange={(e) => setMaxLength(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none accent-indigo-500 cursor-pointer" />
          </div>

          {outputPrompt && (
            <div className="animate-in slide-in-from-bottom-4 duration-500 bg-indigo-950/20 border border-indigo-500/30 rounded-xl p-4 space-y-3">
              <div className="flex justify-between items-center text-[10px] font-bold text-indigo-400 uppercase tracking-widest">
                <span>Результат</span>
                <button onClick={() => navigator.clipboard.writeText(outputPrompt)} className="hover:text-white transition"><Copy size={12} /></button>
              </div>
              <textarea readOnly value={outputPrompt} className="w-full bg-transparent text-xs text-gray-300 font-mono leading-relaxed outline-none h-24 resize-none" />
            </div>
          )}
        </div>
      </aside>

    </div>
  );
};

export default RefinedPromptingService;
