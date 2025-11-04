// Example: src/components/InferenceForm.jsx
import { useState } from 'react';
import axios from 'axios';

function InferenceForm() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const res = await axios.post('/api/infer', { input });
    setResult(res.data);
  };

  return (
    <div>
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={handleSubmit}>Run</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

export default InferenceForm;
