import React, { useState, useEffect } from 'react';
import { RecognitionParams } from '../types';
import { setDetectionParams, getCurrentParams } from '../services/apiService';
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface ParameterControlsProps {
  params: RecognitionParams;
  onChange: (params: RecognitionParams) => void;
}

const ParameterControls: React.FC<ParameterControlsProps> = ({ params, onChange }) => {
  const [localParams, setLocalParams] = useState<RecognitionParams>(params);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncStatus, setSyncStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [syncMessage, setSyncMessage] = useState('');

  // 当 props.params 变化时更新本地状态
  useEffect(() => {
    setLocalParams(params);
  }, [params]);

  // 同步参数到后端
  const syncParamsToBackend = async () => {
    setIsSyncing(true);
    setSyncStatus('idle');
    setSyncMessage('');
    
    try {
      // 只发送变化的参数到后端
      const backendParams: any = {};
      
      if (localParams.minConfidence !== params.minConfidence) {
        backendParams.min_confidence = localParams.minConfidence;
      }
      
      if (localParams.networkSize !== params.networkSize) {
        backendParams.network_size = localParams.networkSize;
      }
      
      if (localParams.minFaceSize !== params.minFaceSize) {
        backendParams.min_face_size = localParams.minFaceSize;
      }
      
      console.log('准备同步参数到后端:', backendParams);
      
      // 如果有变化的参数，则发送到后端
      if (Object.keys(backendParams).length > 0) {
        const result = await setDetectionParams(backendParams);
        setSyncStatus('success');
        setSyncMessage(result.message || '参数更新成功');
        
        // 更新父组件状态
        onChange(localParams);
      } else {
        // 没有变化，直接更新父组件状态
        setSyncStatus('success');
        setSyncMessage('参数无变化，无需更新');
        onChange(localParams);
      }
    } catch (error) {
      console.error('同步参数到后端失败:', error);
      setSyncStatus('error');
      setSyncMessage(error instanceof Error ? error.message : '未知错误');
    } finally {
      setIsSyncing(false);
      
      // 5秒后清除状态提示
      setTimeout(() => {
        setSyncStatus('idle');
        setSyncMessage('');
      }, 5000);
    }
  };

  // 处理参数变化
  const handleParamChange = (paramName: keyof RecognitionParams, value: number) => {
    const newParams = {
      ...localParams,
      [paramName]: value
    };
    setLocalParams(newParams);
  };

  // 重置为默认参数
  const handleResetDefaults = () => {
    const defaultParams: RecognitionParams = {
      minConfidence: 0.5,
      similarityThreshold: 0.6,
      minFaceSize: 20,
      iouThreshold: 0.4,
      maskMode: false,
      networkSize: 640
    };
    setLocalParams(defaultParams);
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-white">参数控制</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={syncParamsToBackend}
            disabled={isSyncing}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors flex items-center gap-1 ${
              isSyncing 
                ? 'bg-blue-700 text-blue-200' 
                : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {isSyncing ? (
              <>
                <span className="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                同步中...
              </>
            ) : '同步参数'}
          </button>
          
          {syncStatus === 'success' && (
            <div className="flex items-center text-green-400 text-sm">
              <CheckCircle className="w-4 h-4 mr-1" />
              {syncMessage || '已同步'}
            </div>
          )}
          
          {syncStatus === 'error' && (
            <div className="flex items-center text-red-400 text-sm">
              <XCircle className="w-4 h-4 mr-1" />
              {syncMessage || '同步失败'}
            </div>
          )}
        </div>
      </div>

      {syncStatus === 'error' && syncMessage && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-200 text-sm">
          <div className="flex items-start">
            <AlertTriangle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
            <div>
              <strong>错误详情:</strong>
              <div className="mt-1">{syncMessage}</div>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {/* 最小置信度 */}
        <div>
          <div className="flex justify-between mb-2">
            <label className="text-slate-200 font-medium">最小置信度</label>
            <span className="text-slate-400 text-sm">{localParams.minConfidence.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="1"
            step="0.01"
            value={localParams.minConfidence}
            onChange={(e) => handleParamChange('minConfidence', parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>更低</span>
            <span>更高</span>
          </div>
          <p className="text-slate-400 text-xs mt-2">
            检测结果的最低可信度阈值。降低此值可检测到更多人脸，但可能增加误检。
          </p>
        </div>

        {/* 网络输入尺寸 */}
        <div>
          <div className="flex justify-between mb-2">
            <label className="text-slate-200 font-medium">网络输入尺寸</label>
            <span className="text-slate-400 text-sm">{localParams.networkSize}px</span>
          </div>
          <input
            type="range"
            min="320"
            max="1024"
            step="32"
            value={localParams.networkSize}
            onChange={(e) => handleParamChange('networkSize', parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>更小</span>
            <span>更大</span>
          </div>
          <p className="text-slate-400 text-xs mt-2">
            输入网络的图像尺寸。较大的尺寸可以提高检测精度，但会增加处理时间。
          </p>
        </div>

        {/* 最小人脸尺寸 */}
        <div>
          <div className="flex justify-between mb-2">
            <label className="text-slate-200 font-medium">最小人脸尺寸</label>
            <span className="text-slate-400 text-sm">{localParams.minFaceSize}px</span>
          </div>
          <input
            type="range"
            min="10"
            max="100"
            step="1"
            value={localParams.minFaceSize}
            onChange={(e) => handleParamChange('minFaceSize', parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>更小</span>
            <span>更大</span>
          </div>
          <p className="text-slate-400 text-xs mt-2">
            能够检测到的最小人脸尺寸。调整此参数可平衡检测小人脸和处理速度。
          </p>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-slate-700">
        <button
          onClick={handleResetDefaults}
          className="w-full py-2 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
        >
          重置为默认参数
        </button>
      </div>
    </div>
  );
};

export default ParameterControls;