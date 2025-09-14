import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Loader2, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface AnalysisResult {
  cleanliness: {
    status: 'CLEAN' | 'DIRTY';
    confidence: number;
  };
  damage: {
    status: 'INTACT' | 'DAMAGED';
    confidence: number;
  };
}

const exampleImages = [
  {
    name: 'Чистый и целый',
    description: 'Идеальное состояние автомобиля',
    url: '/api/placeholder/300/200?text=Clean+Car',
    result: { 
      cleanliness: { status: 'CLEAN' as const, confidence: 96 },
      damage: { status: 'INTACT' as const, confidence: 94 }
    }
  },
  {
    name: 'Грязный, но целый',
    description: 'Нужна мойка',
    url: '/api/placeholder/300/200?text=Dirty+Car',
    result: { 
      cleanliness: { status: 'DIRTY' as const, confidence: 91 },
      damage: { status: 'INTACT' as const, confidence: 97 }
    }
  },
  {
    name: 'Чистый с повреждением',
    description: 'Царапина на кузове',
    url: '/api/placeholder/300/200?text=Scratched+Car',
    result: { 
      cleanliness: { status: 'CLEAN' as const, confidence: 89 },
      damage: { status: 'DAMAGED' as const, confidence: 85 }
    }
  },
  {
    name: 'Грязный и битый',
    description: 'Требует внимания',
    url: '/api/placeholder/300/200?text=Damaged+Dirty+Car',
    result: { 
      cleanliness: { status: 'DIRTY' as const, confidence: 93 },
      damage: { status: 'DAMAGED' as const, confidence: 88 }
    }
  }
];

export default function DemoInterface() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      const imageUrl = URL.createObjectURL(file);
      setUploadedImageUrl(imageUrl);
      setAnalysisResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false
  });

  const simulateAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          // Simulate random result
          setTimeout(() => {
            setAnalysisResult({
              cleanliness: {
                status: Math.random() > 0.5 ? 'CLEAN' : 'DIRTY',
                confidence: Math.floor(Math.random() * 20) + 80
              },
              damage: {
                status: Math.random() > 0.6 ? 'INTACT' : 'DAMAGED',
                confidence: Math.floor(Math.random() * 25) + 75
              }
            });
          }, 500);
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 200);
  };

  const handleExampleClick = (example: typeof exampleImages[0]) => {
    setUploadedImageUrl(example.url);
    setUploadedFile(null);
    setAnalysisResult(null);
    
    // Simulate analysis for example
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          setTimeout(() => {
            setAnalysisResult(example.result);
          }, 500);
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 150);
  };

  const resetDemo = () => {
    setUploadedFile(null);
    setUploadedImageUrl(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    setAnalysisProgress(0);
  };

  const getStatusIcon = (status: string) => {
    if (status === 'CLEAN' || status === 'INTACT') {
      return <CheckCircle className="w-6 h-6 text-indrive-green-500" />;
    }
    return <AlertCircle className="w-6 h-6 text-yellow-500" />;
  };

  const getStatusColor = (status: string) => {
    if (status === 'CLEAN' || status === 'INTACT') {
      return 'text-indrive-green-400';
    }
    return 'text-yellow-400';
  };

  const getStatusText = (category: 'cleanliness' | 'damage', status: string) => {
    if (category === 'cleanliness') {
      return status === 'CLEAN' ? 'ЧИСТЫЙ' : 'ГРЯЗНЫЙ';
    }
    return status === 'INTACT' ? 'ЦЕЛЫЙ' : 'БИТЫЙ';
  };

  return (
    <section id="demo-section" className="py-20 px-6">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="text-gradient">Наше решение</span>{' '}
            <span className="text-white">в действии</span>
          </h2>
          <p className="text-xl text-indrive-green-200 max-w-3xl mx-auto">
            Загрузите фотографию автомобиля и получите мгновенный анализ его состояния
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Upload Area */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>Загрузка изображения</CardTitle>
              <CardDescription>
                Перетащите фото или выберите файл (JPG, PNG, до 10MB)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!uploadedImageUrl ? (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                    isDragActive 
                      ? 'border-indrive-green-400 bg-indrive-green-950/30' 
                      : 'border-indrive-green-600 hover:border-indrive-green-500'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="w-12 h-12 text-indrive-green-400 mx-auto mb-4" />
                  {isDragActive ? (
                    <p className="text-indrive-green-300">Отпустите файл здесь...</p>
                  ) : (
                    <>
                      <p className="text-indrive-green-300 mb-2">
                        Перетащите фото автомобиля сюда
                      </p>
                      <p className="text-sm text-indrive-green-500">
                        или нажмите для выбора файла
                      </p>
                    </>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <img 
                      src={uploadedImageUrl} 
                      alt="Uploaded" 
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    <Button
                      variant="destructive"
                      size="icon"
                      className="absolute top-2 right-2"
                      onClick={resetDemo}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                  
                  {!isAnalyzing && !analysisResult && (
                    <Button 
                      onClick={simulateAnalysis} 
                      className="w-full"
                      size="lg"
                    >
                      Анализировать состояние
                    </Button>
                  )}
                  
                  {isAnalyzing && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-3 text-indrive-green-300">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>AI анализирует состояние... Пожалуйста, подождите</span>
                      </div>
                      <Progress value={analysisProgress} className="w-full" />
                    </div>
                  )}
                  
                  {analysisResult && (
                    <Button 
                      onClick={resetDemo} 
                      variant="outline" 
                      className="w-full"
                      size="lg"
                    >
                      <RotateCcw className="w-4 h-4 mr-2" />
                      Сбросить
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>Результаты анализа</CardTitle>
              <CardDescription>
                Детальная оценка состояния автомобиля
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!analysisResult ? (
                <div className="text-center py-12 text-indrive-green-400">
                  <p>Загрузите изображение для начала анализа</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Cleanliness Result */}
                  <div className="p-4 rounded-lg border border-indrive-green-700 bg-indrive-black-800/50">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        {getStatusIcon(analysisResult.cleanliness.status)}
                        <h3 className="font-semibold text-lg">Чистота</h3>
                      </div>
                      <span className={`text-2xl font-bold ${getStatusColor(analysisResult.cleanliness.status)}`}>
                        {getStatusText('cleanliness', analysisResult.cleanliness.status)}
                      </span>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-indrive-green-300">Уверенность модели:</span>
                        <span className="text-indrive-green-400 font-medium">
                          {analysisResult.cleanliness.confidence}%
                        </span>
                      </div>
                      <Progress value={analysisResult.cleanliness.confidence} />
                    </div>
                  </div>

                  {/* Damage Result */}
                  <div className="p-4 rounded-lg border border-indrive-green-700 bg-indrive-black-800/50">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        {getStatusIcon(analysisResult.damage.status)}
                        <h3 className="font-semibold text-lg">Целостность</h3>
                      </div>
                      <span className={`text-2xl font-bold ${getStatusColor(analysisResult.damage.status)}`}>
                        {getStatusText('damage', analysisResult.damage.status)}
                      </span>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-indrive-green-300">Уверенность модели:</span>
                        <span className="text-indrive-green-400 font-medium">
                          {analysisResult.damage.confidence}%
                        </span>
                      </div>
                      <Progress value={analysisResult.damage.confidence} />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Example Images */}
        <Card>
          <CardHeader>
            <CardTitle>Примеры для тестирования</CardTitle>
            <CardDescription>
              Нажмите на любой пример для быстрого тестирования модели
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {exampleImages.map((example, index) => (
                <div
                  key={index}
                  onClick={() => handleExampleClick(example)}
                  className="cursor-pointer group transition-transform hover:scale-105"
                >
                  <div className="relative overflow-hidden rounded-lg border border-indrive-green-700 hover:border-indrive-green-500">
                    <img 
                      src={example.url} 
                      alt={example.name}
                      className="w-full h-32 object-cover group-hover:brightness-110 transition-all"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                      <div className="absolute bottom-2 left-2 right-2">
                        <p className="text-white text-sm font-medium">{example.name}</p>
                        <p className="text-indrive-green-300 text-xs">{example.description}</p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
