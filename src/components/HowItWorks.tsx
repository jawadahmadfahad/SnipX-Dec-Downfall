import { Upload, Scissors, Type, FastForward, Download, ArrowRight, Sparkles, Play } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

const steps = [
  {
    icon: <Upload className="text-purple-600" size={32} />,
    title: "Upload Your Video",
    description: "Upload your video file to SnipX. We support various formats including MP4, MOV, AVI, and more.",
    color: "purple"
  },
  {
    icon: <Scissors className="text-teal-600" size={32} />,
    title: "Automated Processing",
    description: "Our AI analyzes your video, detects silences and filler words, and prepares it for enhancement.",
    color: "teal"
  },
  {
    icon: <Type className="text-blue-600" size={32} />,
    title: "Add Subtitles",
    description: "Automatically generate Urdu subtitles for your video and customize their appearance.",
    color: "blue"
  },
  {
    icon: <FastForward className="text-orange-500" size={32} />,
    title: "Summarize Content",
    description: "Create a concise summary of your video to highlight the most important parts.",
    color: "orange"
  },
  {
    icon: <Download className="text-green-600" size={32} />,
    title: "Export & Share",
    description: "Export your enhanced video in various formats and share it with your audience.",
    color: "green"
  }
];

const HowItWorks = () => {
  const navigate = useNavigate();
  const [visibleSteps, setVisibleSteps] = useState<number[]>([]);
  const [activeStep, setActiveStep] = useState(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const stepIndex = parseInt(entry.target.getAttribute('data-index') || '0');
            setVisibleSteps(prev => [...prev, stepIndex]);
          }
        });
      },
      { threshold: 0.3 }
    );

    const stepElements = document.querySelectorAll('.step-card');
    stepElements.forEach(step => observer.observe(step));

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (sectionRef.current) {
        const rect = sectionRef.current.getBoundingClientRect();
        setMousePosition({
          x: ((e.clientX - rect.left) / rect.width) * 100,
          y: ((e.clientY - rect.top) / rect.height) * 100,
        });
      }
    };

    const section = sectionRef.current;
    if (section) {
      section.addEventListener('mousemove', handleMouseMove);
      return () => section.removeEventListener('mousemove', handleMouseMove);
    }
  }, []);

  return (
    <section 
      ref={sectionRef}
      id="how-it-works" 
      className="relative py-16 bg-gradient-to-br from-purple-50 via-white to-pink-50 px-4 overflow-hidden"
    >
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div 
          className="absolute w-96 h-96 bg-gradient-to-r from-purple-200/20 to-pink-200/20 rounded-full blur-3xl animate-float"
          style={{
            top: '10%',
            right: '10%',
            transform: `translate(${mousePosition.x * 0.015}px, ${mousePosition.y * 0.015}px)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
        <div 
          className="absolute w-80 h-80 bg-gradient-to-r from-blue-200/20 to-cyan-200/20 rounded-full blur-3xl animate-float-delayed"
          style={{
            bottom: '10%',
            left: '10%',
            transform: `translate(${mousePosition.x * -0.01}px, ${mousePosition.y * -0.01}px)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
        
        {/* Floating Sparkles */}
        <div className="absolute top-20 left-1/4 animate-sparkle">
          <Sparkles className="text-purple-300 w-6 h-6" />
        </div>
        <div className="absolute top-40 right-1/4 animate-sparkle-delayed">
          <Sparkles className="text-pink-300 w-4 h-4" />
        </div>
        <div className="absolute bottom-32 left-1/2 animate-sparkle-slow">
          <Sparkles className="text-blue-300 w-5 h-5" />
        </div>
      </div>

      <div className="container mx-auto max-w-6xl relative z-10">
        <div className="text-center mb-16">
          <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full text-purple-700 text-sm font-medium mb-6 animate-slide-in-up">
            <Play className="w-4 h-4 mr-2" />
            Simple Process
          </div>
          
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 animate-text-reveal">
            How{' '}
            <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              SnipX Works
            </span>
          </h2>
          
          <p className="text-xl text-gray-600 max-w-2xl mx-auto animate-fade-in-up">
            Experience a simple and intuitive workflow designed to streamline your video editing process.
          </p>
        </div>
        
        <div className="relative">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-8 relative z-10">
            {steps.map((step, index) => (
              <div 
                key={index}
                data-index={index}
                className={`step-card group flex flex-col items-center text-center relative transition-all duration-700 ${
                  visibleSteps.includes(index) 
                    ? 'opacity-100 translate-y-0' 
                    : 'opacity-0 translate-y-8'
                } ${
                  activeStep === index ? 'scale-110' : 'scale-100'
                }`}
                style={{ transitionDelay: `${index * 150}ms` }}
              >
                {/* Step Number */}
                <div className={`absolute -top-3 -right-3 w-8 h-8 rounded-full bg-gradient-to-r from-${step.color}-500 to-${step.color}-600 text-white text-sm font-bold flex items-center justify-center shadow-lg z-20 ${
                  activeStep === index ? 'animate-pulse scale-110' : ''
                }`}>
                  {index + 1}
                </div>

                {/* Icon Container */}
                <div className={`relative bg-white rounded-3xl p-6 shadow-xl mb-6 border-2 transition-all duration-500 group-hover:shadow-2xl group-hover:-translate-y-2 ${
                  activeStep === index 
                    ? `border-${step.color}-300 shadow-${step.color}-200/50` 
                    : 'border-gray-100 hover:border-gray-200'
                }`}>
                  <div className={`transform transition-all duration-300 group-hover:scale-110 group-hover:rotate-3 ${
                    activeStep === index ? 'scale-110' : ''
                  }`}>
                    {step.icon}
                  </div>
                  {/* Glow Effect */}
                  <div className={`absolute inset-0 rounded-3xl bg-gradient-to-r from-${step.color}-400/20 to-${step.color}-600/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${
                    activeStep === index ? 'opacity-50' : ''
                  }`}></div>
                </div>

                <h3 className={`text-xl font-bold mb-3 text-gray-900 transition-colors duration-300 ${
                  activeStep === index ? `text-${step.color}-700` : 'group-hover:text-gray-800'
                }`}>
                  {step.title}
                </h3>
                <p className="text-gray-600 group-hover:text-gray-700 transition-colors duration-300 leading-relaxed">
                  {step.description}
                </p>
                {/* Connection Arrow */}
                {index < steps.length - 1 && (
                  <div className={`hidden md:block absolute top-12 -right-8 z-20 transition-all duration-300 ${
                    activeStep >= index ? 'text-purple-500 scale-110' : 'text-gray-300'
                  }`}>
                    <ArrowRight size={24} className="animate-pulse" />
                  </div>
                )}
                {/* Sparkle Effect */}
                {activeStep === index && (
                  <div className="absolute -top-2 -left-2 animate-sparkle">
                    <Sparkles className={`w-4 h-4 text-${step.color}-400`} />
                  </div>
                )}
              </div>
            ))}
          </div>
          {/* Static Progress Line BELOW steps, ABOVE CTA */}
          <div className="hidden md:block w-4/5 mx-auto my-12">
            <div className="h-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full w-full"></div>
          </div>
        </div>
        
        {/* CTA Section */}
        <div className="mt-16 text-center">
          <div 
            className="inline-flex items-center gap-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-4 rounded-2xl shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 cursor-pointer group"
            onClick={() => navigate('/Features')}
          >
            <span className="font-medium text-lg">Get Started Now</span>
            <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center group-hover:bg-white/30 transition-all duration-300 group-hover:rotate-90">
              <ArrowRight className="w-5 h-5 group-hover:translate-x-0.5 transition-transform duration-300" />
            </div>
          </div>
          
          <p className="text-gray-600 mt-4 animate-fade-in-up">
            Join thousands of creators who trust SnipX for their video editing needs
          </p>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;