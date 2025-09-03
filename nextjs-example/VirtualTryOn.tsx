// components/VirtualTryOn.tsx
'use client';

import { useState } from 'react';

interface TryOnResult {
  success?: boolean;
  error?: string;
  result_image?: string;
  mask_image?: string;
}

export default function VirtualTryOn() {
  const [humanImage, setHumanImage] = useState<File | null>(null);
  const [garmentImage, setGarmentImage] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [denoiseSteps, setDenoiseSteps] = useState(60);
  const [seed, setSeed] = useState(-1);
  const [result, setResult] = useState<TryOnResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<string>('');

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860';

  // Health check
  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      const data = await response.json();
      setApiStatus(data.message);
    } catch (error) {
      setApiStatus('API not available');
    }
  };

  // Simple try-on
  const processSimpleTryOn = async () => {
    if (!humanImage || !garmentImage) {
      alert('Please select both human and garment images');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('imgs', humanImage);
      formData.append('garm_img', garmentImage);

      const response = await fetch(`${API_BASE_URL}/api/tryon_simple`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setResult({ success: true, result_image: imageUrl });
      } else {
        const errorData = await response.json();
        setResult({ error: errorData.error || 'Failed to process try-on' });
      }
    } catch (error) {
      setResult({ error: 'Network error occurred' });
    } finally {
      setLoading(false);
    }
  };

  // Try-on with prompt
  const processTryOnWithPrompt = async () => {
    if (!humanImage || !garmentImage) {
      alert('Please select both human and garment images');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('imgs', humanImage);
      formData.append('garm_img', garmentImage);
      formData.append('prompt_text', prompt);

      const response = await fetch(`${API_BASE_URL}/api/tryon_with_prompt`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setResult({ success: true, result_image: imageUrl });
      } else {
        const errorData = await response.json();
        setResult({ error: errorData.error || 'Failed to process try-on' });
      }
    } catch (error) {
      setResult({ error: 'Network error occurred' });
    } finally {
      setLoading(false);
    }
  };

  // Full try-on with all parameters
  const processFullTryOn = async () => {
    if (!humanImage || !garmentImage) {
      alert('Please select both human and garment images');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('imgs', humanImage);
      formData.append('garm_img', garmentImage);
      formData.append('prompt_text', prompt);
      formData.append('denoise_steps', denoiseSteps.toString());
      formData.append('seed', seed.toString());

      const response = await fetch(`${API_BASE_URL}/api/tryon_full`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setResult({ success: true, result_image: imageUrl });
      } else {
        const errorData = await response.json();
        setResult({ error: errorData.error || 'Failed to process try-on' });
      }
    } catch (error) {
      setResult({ error: 'Network error occurred' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">DiffuseFit Virtual Try-On</h1>
      
      {/* API Status */}
      <div className="mb-4">
        <button 
          onClick={checkApiHealth}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Check API Status
        </button>
        {apiStatus && (
          <span className="ml-4 text-sm">
            Status: {apiStatus}
          </span>
        )}
      </div>

      {/* Image Upload */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div>
          <label className="block text-sm font-medium mb-2">
            Human Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setHumanImage(e.target.files?.[0] || null)}
            className="w-full p-2 border border-gray-300 rounded"
          />
          {humanImage && (
            <img
              src={URL.createObjectURL(humanImage)}
              alt="Human"
              className="mt-2 max-w-full h-64 object-cover rounded"
            />
          )}
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Garment Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setGarmentImage(e.target.files?.[0] || null)}
            className="w-full p-2 border border-gray-300 rounded"
          />
          {garmentImage && (
            <img
              src={URL.createObjectURL(garmentImage)}
              alt="Garment"
              className="mt-2 max-w-full h-64 object-cover rounded"
            />
          )}
        </div>
      </div>

      {/* Parameters */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Style Description (Optional)
          </label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., wearing a casual blue t-shirt"
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Denoising Steps: {denoiseSteps}
            </label>
            <input
              type="range"
              min="20"
              max="100"
              value={denoiseSteps}
              onChange={(e) => setDenoiseSteps(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Random Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value))}
              min="-1"
              max="2147483647"
              className="w-full p-2 border border-gray-300 rounded"
            />
            <p className="text-xs text-gray-500 mt-1">
              Use -1 for random results
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-4 mb-6">
        <button
          onClick={processSimpleTryOn}
          disabled={loading}
          className="px-6 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Simple Try-On'}
        </button>

        <button
          onClick={processTryOnWithPrompt}
          disabled={loading}
          className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Try-On with Prompt'}
        </button>

        <button
          onClick={processFullTryOn}
          disabled={loading}
          className="px-6 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Full Try-On'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="border-t pt-6">
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          
          {result.error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              Error: {result.error}
            </div>
          )}

          {result.success && result.result_image && (
            <div>
              <h3 className="text-lg font-medium mb-2">Generated Try-On Image</h3>
              <img
                src={result.result_image}
                alt="Try-on result"
                className="max-w-full h-auto rounded shadow-lg"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
} 