import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Scissors, 
  Type, 
  Volume2, 
  Image, 
  FastForward, 
  Play, 
  Pause, 
  Download, 
  Settings, 
  Sparkles,
  Zap,
  Brain,
  Wand2,
  Eye,
  Star,
  ArrowRight,
  CheckCircle,
  Clock,
  Users,
  TrendingUp
} from 'lucide-react';

const Features = () => {
  const navigate = useNavigate();
  const [activeFeature, setActiveFeature] = useState(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [visibleSections, setVisibleSections] = useState<number[]>([]);
  const [scrollY, setScrollY] = useState(0);
  const featuresRef = useRef<HTMLElement>(null);

  const features = [
    {
      id: 'audio-cutting',
      icon: <Scissors className="text-purple-600" size={32} />,
      title: "Automated Voice Cutting",
      description: "Automatically detect and remove silences, stutters, and filler words to create polished audio.",
      longDescription: "Our advanced AI analyzes your audio waveform to identify and remove unwanted silences, stutters, and filler words like 'um', 'uh', and 'like'. This creates a more professional and engaging listening experience.",
      color: "purple",
      gradient: "from-purple-500 to-indigo-600",
      features: [
        "Smart silence detection",
        "Filler word removal", 
        "Natural flow preservation",
        "Batch processing"
      ],
      stats: { accuracy: "99.2%", timesSaved: "85%", users: "12K+" }
    },
    {
      id: 'subtitling',
      icon: <Type className="text-teal-600" size={32} />,
      title: "Multi-Language Subtitling",
      description: "Generate accurate subtitles in 50+ languages including Urdu, Arabic, and complex scripts.",
      longDescription: "Advanced speech recognition technology supports over 50 languages with special optimization for complex scripts like Urdu, Arabic, and Hindi. Perfect timing synchronization and customizable styling options.",
      color: "teal",
      gradient: "from-teal-500 to-cyan-600",
      features: [
        "50+ languages supported",
        "Complex script handling",
        "Auto-timing sync",
        "Style customization"
      ],
      stats: { accuracy: "96.8%", languages: "50+", users: "25K+" }
    },
    {
      id: 'audio-enhancement',
      icon: <Volume2 className="text-green-600" size={32} />,
      title: "Audio Enhancement",
      description: "Professional-grade audio processing with noise reduction, EQ, and dynamic range optimization.",
      longDescription: "Transform poor audio into broadcast quality with our AI-powered audio enhancement. Includes noise reduction, equalization, dynamic range compression, and voice clarity optimization.",
      color: "green",
      gradient: "from-green-500 to-emerald-600",
      features: [
        "Noise reduction",
        "EQ optimization", 
        "Dynamic range control",
        "Voice clarity boost"
      ],
      stats: { improvement: "300%", processing: "Real-time", users: "18K+" }
    },
    {
      id: 'thumbnail-generation',
      icon: <Image className="text-red-500" size={32} />,
      title: "Smart Thumbnail Generation",
      description: "AI analyzes your video content to generate compelling thumbnails that maximize click-through rates.",
      longDescription: "Our computer vision AI analyzes every frame to identify the most engaging moments for thumbnails. Generate multiple variations with A/B testing insights to maximize your video's click-through rate.",
      color: "red",
      gradient: "from-red-500 to-pink-600",
      features: [
        "AI scene analysis",
        "Multiple variations",
        "Click-rate optimization", 
        "Brand consistency"
      ],
      stats: { clickRate: "+45%", variations: "5+", users: "15K+" }
    },
    {
      id: 'video-summarization',
      icon: <FastForward className="text-orange-500" size={32} />,
      title: "Video Summarization",
      description: "Create concise summaries of long videos to improve engagement and save viewers' time.",
      longDescription: "AI-powered content analysis identifies key moments and creates engaging highlights. Perfect for transforming long-form content into digestible clips that maintain viewer attention.",
      color: "orange",
      gradient: "from-orange-500 to-red-600",
      features: [
        "Key moment detection",
        "Highlight extraction",
        "Custom length control",
        "Engagement optimization"
      ],
      stats: { engagement: "+120%", timeSaved: "70%", users: "22K+" }
    }
  ];

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const sectionIndex = parseInt(entry.target.getAttribute('data-index') || '0');
            setVisibleSections(prev => [...prev, sectionIndex]);
          }
        });
      },
      { threshold: 0.1 }
    );

    const sections = document.querySelectorAll('.feature-section');
    sections.forEach(section => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (featuresRef.current) {
        const rect = featuresRef.current.getBoundingClientRect();
        setMousePosition({
          x: ((e.clientX -rect.left) / rect.width) * 100,
          y: ((e.clientY - rect.top) / rect.height) * 100,
        });
      }
    };

    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    const section = featuresRef.current;
    if (section) {
      section.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('scroll', handleScroll);
      return () => {
        section.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('scroll', handleScroll);
      };
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % features.length);
    }, 9000); // Changed from 4000ms to 9000ms (4s + 5s)

    return () => clearInterval(interval);
  }, []);

  // Function to handle navigation to Features page with specific tab
  const handleTryFeature = (featureId: string) => {
    // Map feature IDs to the corresponding tab names in Features page
    const featureTabMap: { [key: string]: string } = {
      'audio-cutting': 'audio',
      'subtitling': 'subtitles',
      'audio-enhancement': 'enhancement',
      'thumbnail-generation': 'thumbnail',
      'video-summarization': 'summarization'
    };
    
    const tabName = featureTabMap[featureId] || 'audio';
    navigate(`/Features?tab=${tabName}`);
  };

  return (
    <div 
      ref={featuresRef}
      className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 relative overflow-hidden"
      style={{
        transform: `translateY(${scrollY * 0.05}px)`,
      }}
    >
      {/* Enhanced 3D Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Floating 3D Orbs */}
        <div 
          className="absolute w-96 h-96 rounded-full opacity-20"
          style={{
            background: 'radial-gradient(circle, rgba(139, 92, 246, 0.4) 0%, transparent 70%)',
            top: '10%',
            left: '10%',
            transform: `
              translate3d(${mousePosition.x * 0.02}px, ${mousePosition.y * 0.02}px, 0)
              rotateX(${mousePosition.y * 0.1}deg)
              rotateY(${mousePosition.x * 0.1}deg)
            `,
            transition: 'transform 0.3s ease-out',
            animation: 'float-3d 17s ease-in-out infinite' // Changed from 12s to 17s
          }}
        />
        
        <div 
          className="absolute w-80 h-80 rounded-full opacity-15"
          style={{
            background: 'radial-gradient(circle, rgba(236, 72, 153, 0.4) 0%, transparent 70%)',
            top: '60%',
            right: '10%',
            transform: `
              translate3d(${mousePosition.x * -0.015}px, ${mousePosition.y * -0.015}px, 0)
              rotateX(${mousePosition.y * -0.08}deg)
              rotateY(${mousePosition.x * -0.08}deg)
            `,
            transition: 'transform 0.3s ease-out',
            animation: 'float-3d-delayed 20s ease-in-out infinite' // Changed from 15s to 20s
          }}
        />

        {/* 3D Geometric Shapes */}
        <div 
          className="absolute w-32 h-32 opacity-30"
          style={{
            top: '20%',
            right: '20%',
            background: 'linear-gradient(45deg, rgba(59, 130, 246, 0.6), rgba(147, 51, 234, 0.6))',
            transform: `
              translate3d(${mousePosition.x * 0.03}px, ${mousePosition.y * 0.03}px, 0)
              rotateX(${45 + mousePosition.y * 0.2}deg)
              rotateY(${45 + mousePosition.x * 0.2}deg)
              rotateZ(${mousePosition.x * 0.1}deg)
            `,
            transition: 'transform 0.3s ease-out',
            animation: 'spin-3d 25s linear infinite', // Changed from 20s to 25s
            clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)'
          }}
        />

        <div 
          className="absolute w-24 h-24 opacity-40"
          style={{
            bottom: '30%',
            left: '15%',
            background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.6), rgba(59, 130, 246, 0.6))',
            transform: `
              translate3d(${mousePosition.x * -0.02}px, ${mousePosition.y * -0.02}px, 0)
              rotateX(${mousePosition.y * 0.15}deg)
              rotateY(${mousePosition.x * 0.15}deg)
            `,
            transition: 'transform 0.3s ease-out',
            animation: 'bounce-3d 13s ease-in-out infinite', // Changed from 8s to 13s
            borderRadius: '20%'
          }}
        />

        {/* Floating Feature Icons */}
        {features.map((feature, index) => (
          <div
            key={index}
            className="absolute opacity-20"
            style={{
              top: `${15 + index * 15}%`,
              left: `${85 + Math.sin(index) * 10}%`,
              transform: `
                translate3d(${mousePosition.x * (0.01 + index * 0.003)}px, ${mousePosition.y * (0.01 + index * 0.003)}px, 0)
                rotateX(${mousePosition.y * 0.1}deg)
                rotateY(${mousePosition.x * 0.1}deg)
                rotateZ(${index * 72}deg)
              `,
              transition: 'transform 0.3s ease-out',
              animation: `feature-float-${index} ${13 + index * 2}s ease-in-out infinite` // Changed from 8 to 13
            }}
          >
            <div className="w-16 h-16 bg-gradient-to-br from-white/20 to-white/5 rounded-2xl flex items-center justify-center backdrop-blur-sm border border-white/10">
              {feature.icon}
            </div>
          </div>
        ))}

        {/* Particle System */}
        {Array.from({ length: 30 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-purple-400 rounded-full opacity-40"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              transform: `
                translate3d(${mousePosition.x * (0.005 + i * 0.001)}px, ${mousePosition.y * (0.005 + i * 0.001)}px, 0)
              `,
              transition: 'transform 0.3s ease-out',
              animation: `particle-float-${i % 6} ${15 + i * 0.3}s linear infinite` // Changed from 10 to 15
            }}
          />
        ))}
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-20 relative z-10">
        {/* Enhanced Header */}
        <div className="text-center mb-20 animate-slide-up-3d">
          <div 
            className="inline-flex items-center px-6 py-3 bg-white/80 backdrop-blur-md rounded-full shadow-lg border border-white/20 mb-8 transform hover:scale-105 transition-all duration-300 hover:shadow-xl"
            style={{
              transform: `
                perspective(1000px)
                rotateX(${mousePosition.y * 0.05}deg)
                rotateY(${mousePosition.x * 0.05}deg)
                translateZ(20px)
              `,
              transition: 'transform 0.3s ease-out'
            }}
          >
            <Zap className="w-5 h-5 text-purple-600 mr-3 animate-pulse" />
            <span className="text-purple-700 font-medium">Revolutionary AI Features</span>
            <Sparkles className="w-5 h-5 text-pink-600 ml-3 animate-bounce" />
          </div>
          
          <h1 
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
            style={{
              background: 'linear-gradient(135deg, #1f2937 0%, #8b5cf6 25%, #ec4899 50%, #3b82f6 75%, #1f2937 100%)',
              backgroundSize: '300% 300%',
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              animation: 'gradient-text-flow 4s ease-in-out infinite',
              textShadow: '0 0 40px rgba(139, 92, 246, 0.3)',
              transform: `
                perspective(1000px)
                rotateX(${mousePosition.y * 0.02}deg)
                rotateY(${mousePosition.x * 0.02}deg)
                translateZ(30px)
              `,
              transition: 'transform 0.3s ease-out'
            }}
          >
            AI-Powered
            <br />
            <span className="relative">
              Features
              <div className="absolute -inset-4 bg-gradient-to-r from-purple-600/20 to-pink-600/20 blur-2xl rounded-full animate-pulse-glow" />
            </span>
          </h1>
          
          <p 
            className="text-xl md:text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed"
            style={{
              transform: `
                perspective(1000px)
                rotateX(${mousePosition.y * 0.01}deg)
                rotateY(${mousePosition.x * 0.01}deg)
                translateZ(15px)
              `,
              transition: 'transform 0.3s ease-out'
            }}
          >
            Experience the future of video editing with cutting-edge AI that transforms 
            your creative vision into stunning reality with unprecedented speed and precision.
          </p>
        </div>

        {/* Enhanced Feature Navigation */}
        <div 
          className="flex justify-center mb-16"
          style={{
            transform: `
              perspective(1000px)
              rotateX(${mousePosition.y * 0.01}deg)
              rotateY(${mousePosition.x * 0.01}deg)
              translateZ(10px)
            `,
            transition: 'transform 0.3s ease-out'
          }}
        >
          <div className="flex space-x-2 bg-white/80 backdrop-blur-md rounded-2xl p-2 border border-white/20 shadow-xl">
            {features.map((feature, index) => (
              <button
                key={feature.id}
                onClick={() => setActiveFeature(index)}
                className={`flex items-center px-6 py-3 rounded-xl transition-all duration-300 transform hover:scale-105 ${
                  activeFeature === index
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                }`}
                style={{
                  transform: activeFeature === index 
                    ? 'translateZ(10px) rotateY(2deg)' 
                    : 'translateZ(0px)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                }}
              >
                <div className="mr-3 transform transition-all duration-300" style={{
                  transform: activeFeature === index ? 'scale(1.1) rotateZ(5deg)' : 'scale(1)'
                }}>
                  {feature.icon}
                </div>
                <span className="font-medium hidden md:block">{feature.title}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Enhanced Active Feature Display */}
        <div className="max-w-6xl mx-auto relative" style={{ height: '800px' }}>
          {features.map((feature, index) => (
            <div
              key={feature.id}
              data-index={index}
              className="feature-section absolute inset-0 transition-all duration-1000"
              style={{
                opacity: activeFeature === index ? 1 : 0,
                visibility: activeFeature === index ? 'visible' : 'hidden',
                transform: activeFeature === index 
                  ? `
                    perspective(1000px)
                    rotateX(${mousePosition.y * 0.005}deg)
                    rotateY(${mousePosition.x * 0.005}deg)
                    translateZ(20px)
                    scale(1)
                  `
                  : `
                    perspective(1000px)
                    translateZ(-50px)
                    scale(0.95)
                  `,
                transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1), visibility 0s linear ' + 
                  (activeFeature === index ? '0s' : '0.8s')
              }}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                {/* Enhanced Feature Content */}
                <div className="space-y-8">
                  <div 
                    className="space-y-6"
                    style={{
                      animation: activeFeature === index ? 'slide-in-left-3d 0.8s ease-out' : 'none',
                      opacity: activeFeature === index ? 1 : 0
                    }}
                  >
                    <div className="flex items-center space-x-4">
                      <div 
                        className={`p-4 rounded-2xl bg-gradient-to-br from-${feature.color}-100 to-${feature.color}-200 shadow-2xl border border-${feature.color}-200`}
                        style={{
                          transform: `
                            rotateY(${mousePosition.x * 0.1}deg)
                            rotateX(${mousePosition.y * 0.1}deg)
                            translateZ(15px)
                          `,
                          transition: 'transform 0.3s ease-out'
                        }}
                      >
                        <div className="transform transition-all duration-300 hover:scale-110 hover:rotateZ(5deg)">
                          {feature.icon}
                        </div>
                        <div className={`absolute inset-0 bg-${feature.color}-400/20 rounded-2xl blur-xl opacity-0 hover:opacity-100 transition-opacity duration-300 animate-pulse-glow`} />
                      </div>
                      
                      <div>
                        <h2 className="text-4xl font-bold text-gray-900 mb-2">
                          {feature.title}
                        </h2>
                        <p className="text-gray-600 text-lg">
                          {feature.description}
                        </p>
                      </div>
                    </div>

                    <p className="text-gray-700 text-lg leading-relaxed">
                      {feature.longDescription}
                    </p>

                    {/* Enhanced Feature Stats */}
                    <div className="grid grid-cols-3 gap-6">
                      {Object.entries(feature.stats).map(([key, value], statIndex) => (
                        <div 
                          key={key}
                          className="text-center bg-white/80 backdrop-blur-sm rounded-2xl p-4 shadow-lg border border-white/20 transform hover:scale-105 transition-all duration-300"
                          style={{
                            animation: activeFeature === index ? `stat-reveal 0.6s ease-out ${statIndex * 200}ms both` : 'none'
                          }}
                        >
                          <div 
                            className="text-3xl font-bold text-gray-900 mb-1"
                            style={{
                              background: `linear-gradient(135deg, #1f2937, #${feature.color === 'purple' ? '8b5cf6' : feature.color === 'teal' ? '14b8a6' : feature.color === 'green' ? '10b981' : feature.color === 'red' ? 'ef4444' : 'f97316'})`,
                              WebkitBackgroundClip: 'text',
                              backgroundClip: 'text',
                              WebkitTextFillColor: 'transparent'
                            }}
                          >
                            {value}
                          </div>
                          <div className="text-gray-500 text-sm capitalize">
                            {key.replace(/([A-Z])/g, ' $1').trim()}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Enhanced Feature List */}
                    <div className="space-y-3">
                      {feature.features.map((item, itemIndex) => (
                        <div 
                          key={item}
                          className="flex items-center space-x-3 bg-white/60 backdrop-blur-sm rounded-xl p-3 shadow-md border border-white/30 transform hover:scale-105 transition-all duration-300"
                          style={{
                            animation: activeFeature === index ? `feature-item-reveal 0.5s ease-out ${itemIndex * 100}ms both` : 'none'
                          }}
                        >
                          <div className={`w-3 h-3 bg-${feature.color}-500 rounded-full animate-pulse`} />
                          <span className="text-gray-700 font-medium">{item}</span>
                          <CheckCircle className={`w-4 h-4 text-${feature.color}-500`} />
                        </div>
                      ))}
                    </div>

                    {/* Enhanced Action Buttons */}
                    <div 
                      className="flex space-x-4"
                      style={{
                        animation: activeFeature === index ? 'button-reveal 0.8s ease-out 0.6s both' : 'none'
                      }}
                    >
                      <button 
                        className={`px-8 py-4 bg-gradient-to-r ${feature.gradient} text-white rounded-2xl font-semibold hover:shadow-2xl transition-all duration-300 transform hover:scale-105 flex items-center`}
                        style={{
                          boxShadow: `0 10px 30px rgba(139, 92, 246, 0.4)`
                        }}
                        onClick={() => handleTryFeature(feature.id)}
                      >
                        <Play size={20} className="mr-2" />
                        Try {feature.title}
                      </button>
                      
                      <button 
                        className="px-6 py-4 bg-white/80 backdrop-blur-md text-gray-700 rounded-2xl font-medium hover:bg-white/90 transition-all duration-300 transform hover:scale-105 border border-gray-200 flex items-center shadow-lg"
                        onClick={() => navigate('/Help')}
                      >
                        <Eye size={20} className="mr-2" />
                        Learn More
                      </button>
                    </div>
                  </div>
                </div>

                {/* Enhanced Feature Demo/Visualization */}
                <div 
                  className="relative"
                  style={{
                    animation: activeFeature === index ? 'slide-in-right-3d 0.8s ease-out 0.3s both' : 'none',
                    opacity: activeFeature === index ? 1 : 0,
                    visibility: activeFeature === index ? 'visible' : 'hidden',
                    transition: 'opacity 0.8s ease-out, visibility 0s ' + 
                      (activeFeature === index ? '0s' : '0.8s')
                  }}
                >
                  <div 
                    className="relative bg-white/90 backdrop-blur-md rounded-3xl p-8 border border-white/30 shadow-2xl"
                    style={{
                      transform: `
                        perspective(1000px)
                        rotateY(${mousePosition.x * 0.02}deg)
                        rotateX(${mousePosition.y * 0.02}deg)
                        translateZ(25px)
                      `,
                      transition: 'transform 0.3s ease-out'
                    }}
                  >
                    {/* Feature-specific Demo Content */}
                    {feature.id === 'audio-cutting' && (
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                          <Wand2 className="mr-2 text-purple-600" size={24} />
                          Audio Waveform Analysis
                        </h3>
                        
                        {/* Animated Waveform */}
                        <div className="bg-gray-100 rounded-xl p-6 border border-gray-200">
                          <div className="flex items-end justify-center space-x-1 h-24">
                            {Array.from({ length: 40 }).map((_, i) => (
                              <div
                                key={i}
                                className={`bg-gradient-to-t ${feature.gradient} rounded-t-sm transition-all duration-300`}
                                style={{
                                  width: '4px',
                                  height: `${20 + Math.sin(i * 0.3 + Date.now() * 0.003) * 30}px`,
                                  animation: `waveform-${i % 4} 2s ease-in-out infinite ${i * 50}ms`
                                }}
                              />
                            ))}
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-2">
                            <span>0:00</span>
                            <span className="text-purple-600 font-medium">Silence Detected</span>
                            <span>2:45</span>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-red-100 border border-red-200 rounded-xl p-4 text-center">
                            <div className="text-red-600 text-2xl font-bold">45%</div>
                            <div className="text-red-500 text-sm">Silence Removed</div>
                          </div>
                          <div className="bg-green-100 border border-green-200 rounded-xl p-4 text-center">
                            <div className="text-green-600 text-2xl font-bold">2.3x</div>
                            <div className="text-green-500 text-sm">Faster Playback</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {feature.id === 'subtitling' && (
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                          <Brain className="mr-2 text-teal-600" size={24} />
                          Multi-Language Processing
                        </h3>
                        
                        <div className="space-y-4">
                          {[
                            { lang: 'English', text: 'Welcome to our video demonstration', flag: '🇺🇸' },
                            { lang: 'Urdu', text: 'ہمارے ویڈیو ڈیمونسٹریشن میں خوش آمدید', flag: '🇵🇰' },
                            { lang: 'Arabic', text: 'مرحباً بكم في عرضنا التوضيحي', flag: '🇸🇦' }
                          ].map((subtitle, i) => (
                            <div 
                              key={i}
                              className="bg-gray-100 rounded-xl p-4 border border-gray-200 transform hover:scale-105 transition-all duration-300"
                              style={{
                                animation: `subtitle-appear ${0.5}s ease-out ${i * 200}ms both`
                              }}
                            >
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-teal-600 text-sm font-medium flex items-center">
                                  <span className="mr-2">{subtitle.flag}</span>
                                  {subtitle.lang}
                                </span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                  <span className="text-green-600 text-xs">Live</span>
                                </div>
                              </div>
                              <p className="text-gray-800 text-sm">{subtitle.text}</p>
                            </div>
                          ))}
                        </div>

                        <div className="bg-teal-100 border border-teal-200 rounded-xl p-4 text-center">
                          <div className="text-teal-600 text-2xl font-bold">50+</div>
                          <div className="text-teal-500 text-sm">Languages Supported</div>
                        </div>
                      </div>
                    )}

                    {feature.id === 'audio-enhancement' && (
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                          <Volume2 className="mr-2 text-green-600" size={24} />
                          Real-time Audio Processing
                        </h3>
                        
                        {/* Audio Enhancement Controls */}
                        <div className="space-y-4">
                          {[
                            { label: 'Noise Reduction', value: 85, color: 'green' },
                            { label: 'Voice Clarity', value: 92, color: 'blue' },
                            { label: 'Dynamic Range', value: 78, color: 'purple' }
                          ].map((control, i) => (
                            <div key={control.label} className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-700">{control.label}</span>
                                <span className={`text-${control.color}-600 font-medium`}>{control.value}%</span>
                              </div>
                              <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
                                <div 
                                  className={`bg-gradient-to-r from-${control.color}-500 to-${control.color}-400 h-2 rounded-full transition-all duration-1000`}
                                  style={{ 
                                    width: `${control.value}%`,
                                    animation: `progress-fill 2s ease-out ${i * 300}ms both`
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Before/After Comparison */}
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-red-100 border border-red-200 rounded-xl p-4 text-center">
                            <div className="text-red-600 text-lg font-bold">Before</div>
                            <div className="text-red-500 text-sm">Noisy Audio</div>
                            <div className="mt-2 flex justify-center">
                              <div className="flex space-x-1">
                                {Array.from({ length: 8 }).map((_, i) => (
                                  <div
                                    key={i}
                                    className="w-1 bg-red-500 rounded-full"
                                    style={{
                                      height: `${10 + Math.random() * 20}px`,
                                      animation: `noise-wave 1s ease-in-out infinite ${i * 100}ms`
                                    }}
                                  />
                                ))}
                              </div>
                            </div>
                          </div>
                          
                          <div className="bg-green-100 border border-green-200 rounded-xl p-4 text-center">
                            <div className="text-green-600 text-lg font-bold">After</div>
                            <div className="text-green-500 text-sm">Crystal Clear</div>
                            <div className="mt-2 flex justify-center">
                              <div className="flex space-x-1">
                                {Array.from({ length: 8 }).map((_, i) => (
                                  <div
                                    key={i}
                                    className="w-1 bg-green-500 rounded-full"
                                    style={{
                                      height: `${15 + Math.sin(i * 0.5) * 10}px`,
                                      animation: `clean-wave 1.5s ease-in-out infinite ${i * 150}ms`
                                    }}
                                  />
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {feature.id === 'thumbnail-generation' && (
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                          <Eye className="mr-2 text-red-600" size={24} />
                          AI Thumbnail Analysis
                        </h3>
                        
                        {/* Thumbnail Grid */}
                        <div className="grid grid-cols-2 gap-4">
                          {[
                            'https://images.pexels.com/photos/2399840/pexels-photo-2399840.jpeg?auto=compress&cs=tinysrgb&w=300&h=200',
                            'https://images.pexels.com/photos/3945313/pexels-photo-3945313.jpeg?auto=compress&cs=tinysrgb&w=300&h=200',
                            'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?auto=compress&cs=tinysrgb&w=300&h=200',
                            'https://images.pexels.com/photos/2608517/pexels-photo-2608517.jpeg?auto=compress&cs=tinysrgb&w=300&h=200'
                          ].map((thumb, i) => (
                            <div 
                              key={i}
                              className="relative group cursor-pointer rounded-xl overflow-hidden border-2 border-transparent hover:border-red-400 transition-all duration-300 transform hover:scale-105"
                              style={{
                                animation: `thumbnail-reveal 0.6s ease-out ${i * 150}ms both`,
                                transform: `perspective(500px) rotateY(${i % 2 === 0 ? -2 : 2}deg)`
                              }}
                            >
                              <img 
                                src={thumb} 
                                alt={`AI Generated Thumbnail ${i + 1}`}
                                className="w-full h-24 object-cover group-hover:scale-110 transition-transform duration-300"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                              <div className="absolute bottom-2 left-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                <div className="flex items-center justify-between">
                                  <span className="text-white text-xs font-medium">AI Score: {85 + i * 3}%</span>
                                  <Star className="w-3 h-3 text-yellow-400 fill-current" />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>

                        <div className="bg-red-100 border border-red-200 rounded-xl p-4 text-center">
                          <div className="text-red-600 text-2xl font-bold">+45%</div>
                          <div className="text-red-500 text-sm">Click-through Rate Increase</div>
                        </div>
                      </div>
                    )}

                    {feature.id === 'video-summarization' && (
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                          <FastForward className="mr-2 text-orange-600" size={24} />
                          Content Intelligence
                        </h3>
                        
                        {/* Timeline Visualization */}
                        <div className="bg-gray-100 rounded-xl p-6 border border-gray-200">
                          <div className="space-y-4">
                            <div className="flex justify-between text-sm text-gray-600">
                              <span>Original: 15:30</span>
                              <span className="text-orange-600 font-medium">Summary: 4:45</span>
                            </div>
                            
                            {/* Timeline Bars */}
                            <div className="space-y-3">
                              <div className="bg-gray-300 rounded-full h-3 overflow-hidden">
                                <div className="bg-gradient-to-r from-gray-500 to-gray-600 h-3 rounded-full w-full" />
                              </div>
                              <div className="bg-gray-300 rounded-full h-3 overflow-hidden">
                                <div 
                                  className="bg-gradient-to-r from-orange-500 to-red-500 h-3 rounded-full transition-all duration-2000"
                                  style={{ 
                                    width: '30%',
                                    animation: 'summary-progress 3s ease-out'
                                  }}
                                />
                              </div>
                            </div>

                            {/* Key Moments */}
                            <div className="flex justify-between text-xs text-gray-500">
                              <span>Intro</span>
                              <span className="text-orange-600 font-medium">Key Point 1</span>
                              <span className="text-orange-600 font-medium">Key Point 2</span>
                              <span>Conclusion</span>
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-orange-100 border border-orange-200 rounded-xl p-4 text-center">
                            <div className="text-orange-600 text-2xl font-bold">70%</div>
                            <div className="text-orange-500 text-sm">Time Saved</div>
                          </div>
                          <div className="bg-blue-100 border border-blue-200 rounded-xl p-4 text-center">
                            <div className="text-blue-600 text-2xl font-bold">+120%</div>
                            <div className="text-blue-500 text-sm">Engagement</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Floating Elements */}
                  <div className="absolute -top-4 -right-4 opacity-60">
                    <div 
                      className={`w-16 h-16 bg-gradient-to-br from-${feature.color}-400/30 to-${feature.color}-600/30 rounded-full`}
                      style={{
                        animation: 'float-decoration 6s ease-in-out infinite',
                        transform: `rotateZ(${mousePosition.x * 0.1}deg)`
                      }}
                    />
                  </div>
                  
                  <div className="absolute -bottom-6 -left-6 opacity-40">
                    <div 
                      className={`w-24 h-24 bg-gradient-to-br from-${feature.color}-500/20 to-${feature.color}-700/20 rounded-full blur-xl`}
                      style={{
                        animation: 'float-decoration-delayed 8s ease-in-out infinite',
                        transform: `rotateZ(-${mousePosition.y * 0.1}deg)`
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Enhanced Feature Navigation Dots */}
        <div className="flex justify-center mt-16 space-x-4">
          {features.map((_, index) => (
            <button
              key={index}
              onClick={() => setActiveFeature(index)}
              className={`w-3 h-3 rounded-full transition-all duration-300 transform hover:scale-125 ${
                activeFeature === index 
                  ? 'bg-purple-600 shadow-lg' 
                  : 'bg-gray-300 hover:bg-gray-400'
              }`}
              style={{
                animation: activeFeature === index ? 'dot-pulse 2s ease-in-out infinite' : 'none'
              }}
            />
          ))}
        </div>
      </div>

      {/* Custom Styles */}
      <style>{`
        @keyframes gradient-text-flow {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        @keyframes pulse-glow {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.05); }
        }
        
        @keyframes slide-in-left-3d {
          0% { 
            transform: translateX(-20px) translateZ(-20px) rotateY(-5deg); 
            opacity: 0; 
          }
          100% { 
            transform: translateX(0px) translateZ(0px) rotateY(0deg); 
            opacity: 1; 
          }
        }
        
        @keyframes slide-in-right-3d {
          0% { 
            transform: translateX(20px) translateZ(-20px) rotateY(5deg); 
            opacity: 0; 
          }
          100% { 
            transform: translateX(0px) translateZ(0px) rotateY(0deg); 
            opacity: 1; 
          }
        }
        
        @keyframes stat-reveal {
          0% { 
            transform: translateY(20px) rotateX(-10deg); 
            opacity: 0; 
          }
          100% { 
            transform: translateY(0px) rotateX(0deg); 
            opacity: 1; 
          }
        }
        
        @keyframes feature-item-reveal {
          0% { 
            transform: translateX(-20px) rotateY(-5deg); 
            opacity: 0; 
          }
          100% { 
            transform: translateX(0px) rotateY(0deg); 
            opacity: 1; 
          }
        }
        
        @keyframes button-reveal {
          0% { 
            transform: translateY(30px) rotateX(-15deg) scale(0.9); 
            opacity: 0; 
          }
          100% { 
            transform: translateY(0px) rotateX(0deg) scale(1); 
            opacity: 1; 
          }
        }
        
        @keyframes waveform-0 {
          0%, 100% { height: 20px; }
          50% { height: 40px; }
        }
        
        @keyframes waveform-1 {
          0%, 100% { height: 15px; }
          50% { height: 35px; }
        }
        
        @keyframes waveform-2 {
          0%, 100% { height: 25px; }
          50% { height: 45px; }
        }
        
        @keyframes waveform-3 {
          0%, 100% { height: 18px; }
          50% { height: 38px; }
        }
        
        @keyframes subtitle-appear {
          0% { 
            transform: translateY(20px) rotateX(-10deg); 
            opacity: 0; 
          }
          100% { 
            transform: translateY(0px) rotateX(0deg); 
            opacity: 1; 
          }
        }
        
        @keyframes progress-fill {
          0% { width: 0%; }
          100% { width: var(--target-width); }
        }
        
        @keyframes noise-wave {
          0%, 100% { height: 10px; opacity: 0.6; }
          50% { height: 30px; opacity: 1; }
        }
        
        @keyframes clean-wave {
          0%, 100% { height: 15px; opacity: 0.8; }
          50% { height: 25px; opacity: 1; }
        }
        
        @keyframes thumbnail-reveal {
          0% { 
            transform: perspective(500px) rotateY(45deg) translateZ(-20px); 
            opacity: 0; 
          }
          100% { 
            transform: perspective(500px) rotateY(0deg) translateZ(0px); 
            opacity: 1; 
          }
        }
        
        @keyframes summary-progress {
          0% { width: 0%; }
          100% { width: 30%; }
        }
        
        @keyframes float-decoration {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(-10px) rotateZ(180deg); }
        }
        
        @keyframes float-decoration-delayed {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(8px) rotateZ(-180deg); }
        }
        
        @keyframes dot-pulse {
          0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.7); }
          50% { transform: scale(1.2); box-shadow: 0 0 0 10px rgba(139, 92, 246, 0); }
        }
        
        @keyframes gradient-flow {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        @keyframes feature-float-0 {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(-20px) rotateZ(180deg); }
        }
        
        @keyframes feature-float-1 {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(-15px) rotateZ(-180deg); }
        }
        
        @keyframes feature-float-2 {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(-25px) rotateZ(90deg); }
        }
        
        @keyframes feature-float-3 {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(-18px) rotateZ(-90deg); }
        }
        
        @keyframes feature-float-4 {
          0%, 100% { transform: translateY(0px) rotateZ(0deg); }
          50% { transform: translateY(-22px) rotateZ(270deg); }
        }
        
        @keyframes particle-float-0 {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.4; }
          50% { transform: translateY(-100px) translateX(20px); opacity: 0.8; }
        }
        
        @keyframes particle-float-1 {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
          50% { transform: translateY(-80px) translateX(-15px); opacity: 0.7; }
        }
        
        @keyframes particle-float-2 {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.5; }
          50% { transform: translateY(-120px) translateX(10px); opacity: 0.9; }
        }
        
        @keyframes particle-float-3 {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.4; }
          50% { transform: translateY(-90px) translateX(-25px); opacity: 0.6; }
        }
        
        @keyframes particle-float-4 {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
          50% { transform: translateY(-110px) translateX(30px); opacity: 0.8; }
        }
        
        @keyframes particle-float-5 {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.4; }
          50% { transform: translateY(-95px) translateX(-20px); opacity: 0.7; }
        }
        
        @keyframes float-3d {
          0%, 100% { transform: translate3d(0px, 0px, 0px) rotateY(0deg); }
          50% { transform: translate3d(20px, -20px, 10px) rotateY(180deg); }
        }
        
        @keyframes float-3d-delayed {
          0%, 100% { transform: translate3d(0px, 0px, 0px) rotateY(0deg); }
          50% { transform: translate3d(-15px, 15px, -10px) rotateY(-180deg); }
        }
        
        @keyframes spin-3d {
          0% { transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }
          33% { transform: rotateX(120deg) rotateY(120deg) rotateZ(0deg); }
          66% { transform: rotateX(240deg) rotateY(240deg) rotateZ(120deg); }
          100% { transform: rotateX(360deg) rotateY(360deg) rotateZ(360deg); }
        }
        
        @keyframes bounce-3d {
          0%, 100% { transform: translate3d(0px, 0px, 0px) scale(1); }
          50% { transform: translate3d(0px, -30px, 20px) scale(1.1); }
        }
      `}</style>
    </div>
  );
};

export default Features;