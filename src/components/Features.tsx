import { FilmIcon, Type, FileText, Image, Scissors, Volume2, FastForward, Download, Sparkles, Zap } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';

const features = [
  {
    icon: <Scissors className="text-purple-600" size={32} />,
    title: "Automated Voice Cutting",
    description: "Automatically detect and remove silences, stutters, and filler words to create polished audio.",
    color: "purple"
  },
  {
    icon: <Type className="text-teal-600" size={32} />,
    title: "Urdu Subtitling",
    description: "Generate accurate Urdu subtitles automatically. Edit and style them to match your preferences.",
    color: "teal"
  },
  {
    icon: <FastForward className="text-orange-500" size={32} />,
    title: "Video Summarization",
    description: "Create concise summaries of long videos to improve engagement and save viewers' time.",
    color: "orange"
  },
  {
    icon: <Volume2 className="text-green-600" size={32} />,
    title: "Audio Enhancement",
    description: "Improve audio quality with noise reduction, equalization, and volume normalization.",
    color: "green"
  },
  {
    icon: <FilmIcon className="text-blue-600" size={32} />,
    title: "Visual Enhancement",
    description: "Stabilize shaky footage, adjust colors, and improve overall visual quality.",
    color: "blue"
  },
  {
    icon: <Image className="text-red-500" size={32} />,
    title: "Thumbnail Generation",
    description: "Create eye-catching thumbnails to attract more viewers to your content.",
    color: "red"
  },
  {
    icon: <FileText className="text-indigo-600" size={32} />,
    title: "Multi-format Support",
    description: "Import and export videos in various formats to suit your specific needs.",
    color: "indigo"
  },
  {
    icon: <Download className="text-gray-700" size={32} />,
    title: "Easy Export",
    description: "Export your enhanced videos in multiple quality settings for different platforms.",
    color: "gray"
  }
];

const Features = () => {
  const [visibleCards, setVisibleCards] = useState<number[]>([]);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const cardIndex = parseInt(entry.target.getAttribute('data-index') || '0');
            setVisibleCards(prev => [...prev, cardIndex]);
          }
        });
      },
      { threshold: 0.1 }
    );

    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => observer.observe(card));

    return () => observer.disconnect();
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
      id="features" 
      className="relative py-16 bg-gradient-to-br from-gray-50 via-white to-purple-50 px-4 overflow-hidden"
    >
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div 
          className="absolute w-96 h-96 bg-gradient-to-r from-purple-300/10 to-pink-300/10 rounded-full blur-3xl animate-float"
          style={{
            top: '20%',
            left: '80%',
            transform: `translate(${mousePosition.x * 0.01}px, ${mousePosition.y * 0.01}px)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
        <div 
          className="absolute w-80 h-80 bg-gradient-to-r from-blue-300/10 to-cyan-300/10 rounded-full blur-3xl animate-float-delayed"
          style={{
            bottom: '20%',
            left: '10%',
            transform: `translate(${mousePosition.x * -0.008}px, ${mousePosition.y * -0.008}px)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
      </div>

      <div className="container mx-auto max-w-6xl relative z-10">
        <div className="text-center mb-16">
          <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full text-purple-700 text-sm font-medium mb-6 animate-slide-in-up">
            <Zap className="w-4 h-4 mr-2" />
            Powerful AI Features
          </div>
          
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 animate-text-reveal">
            Transform Your Videos with{' '}
            <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              AI Magic
            </span>
          </h2>
          
          <p className="text-xl text-gray-600 max-w-2xl mx-auto animate-fade-in-up">
            Discover how SnipX can transform your video editing workflow and help you create
            professional content with minimal effort.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              data-index={index}
              className={`feature-card group bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all duration-500 border border-white/20 hover:border-${feature.color}-200/50 transform hover:-translate-y-2 hover:scale-105 ${
                visibleCards.includes(index) 
                  ? 'opacity-100 translate-y-0' 
                  : 'opacity-0 translate-y-8'
              }`}
              style={{
                transitionDelay: `${index * 100}ms`,
                background: `linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)`
              }}
            >
              {/* Icon Container with Glow Effect */}
              <div className={`relative mb-4 p-3 rounded-2xl bg-gradient-to-br from-${feature.color}-50 to-${feature.color}-100 group-hover:from-${feature.color}-100 group-hover:to-${feature.color}-200 transition-all duration-300`}>
                <div className="relative z-10 transform group-hover:scale-110 group-hover:rotate-3 transition-all duration-300">
                  {feature.icon}
                </div>
                <div className={`absolute inset-0 bg-${feature.color}-400/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div>
              </div>

              <h3 className="text-xl font-bold mb-3 text-gray-900 group-hover:text-gray-800 transition-colors duration-300">
                {feature.title}
              </h3>
              
              <p className="text-gray-600 group-hover:text-gray-700 transition-colors duration-300 leading-relaxed">
                {feature.description}
              </p>

              {/* Hover Effect Overlay */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-transparent via-transparent to-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
              
              {/* Sparkle Effect on Hover */}
              <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-all duration-300 transform group-hover:scale-110">
                <Sparkles className={`w-5 h-5 text-${feature.color}-400 animate-pulse`} />
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA Section */}
        <div className="text-center mt-16">
          <div className="inline-flex items-center gap-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-4 rounded-2xl shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 cursor-pointer group">
            <span className="font-medium">Ready to get started?</span>
            <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center group-hover:bg-white/30 transition-colors duration-300">
              <FastForward className="w-4 h-4 group-hover:translate-x-0.5 transition-transform duration-300" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;