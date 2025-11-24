import { 
  Code2, 
  Database, 
  Server, 
  Cpu, 
  Cloud, 
  Lock,
  Layers,
  GitBranch,
  Workflow,
  Zap,
  Sparkles
} from 'lucide-react';
import { useEffect, useState } from 'react';

const Technologies = () => {
  const [visibleSections, setVisibleSections] = useState<number[]>([]);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

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

    const sections = document.querySelectorAll('.tech-section');
    sections.forEach(section => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const techStacks = [
    {
      category: "Frontend Technologies",
      items: [
        {
          icon: <Code2 className="text-blue-500" size={32} />,
          name: "React",
          description: "Component-based UI library for building interactive interfaces",
          features: ["Component Architecture", "Virtual DOM", "React Hooks", "Context API"],
          color: "blue"
        },
        {
          icon: <Layers className="text-purple-500" size={32} />,
          name: "Tailwind CSS",
          description: "Utility-first CSS framework for rapid UI development",
          features: ["Responsive Design", "Custom Components", "Dark Mode", "Animations"],
          color: "purple"
        }
      ]
    },
    {
      category: "Backend Technologies",
      items: [
        {
          icon: <Server className="text-green-500" size={32} />,
          name: "Flask",
          description: "Lightweight Python web framework for building APIs",
          features: ["RESTful APIs", "JWT Authentication", "Middleware Support", "Database Integration"],
          color: "green"
        },
        {
          icon: <Database className="text-yellow-500" size={32} />,
          name: "MongoDB",
          description: "NoSQL database for flexible data storage",
          features: ["Document Storage", "Scalability", "Rich Queries", "Real-time Updates"],
          color: "yellow"
        }
      ]
    },
    {
      category: "AI & Processing",
      items: [
        {
          icon: <Cpu className="text-red-500" size={32} />,
          name: "TensorFlow",
          description: "Machine learning framework for video processing",
          features: ["Video Analysis", "Speech Recognition", "Object Detection", "Scene Understanding"],
          color: "red"
        },
        {
          icon: <Workflow className="text-indigo-500" size={32} />,
          name: "FFmpeg",
          description: "Multimedia framework for video manipulation",
          features: ["Video Cutting", "Format Conversion", "Stream Processing", "Audio Extraction"],
          color: "indigo"
        }
      ]
    },
    {
      category: "Infrastructure",
      items: [
        {
          icon: <Cloud className="text-sky-500" size={32} />,
          name: "Cloud Services",
          description: "Scalable cloud infrastructure for video processing",
          features: ["Auto Scaling", "Load Balancing", "CDN Integration", "Storage Solutions"],
          color: "sky"
        },
        {
          icon: <Lock className="text-gray-500" size={32} />,
          name: "Security",
          description: "Comprehensive security measures for data protection",
          features: ["Encryption", "Access Control", "Secure Storage", "Compliance"],
          color: "gray"
        }
      ]
    },
    {
      category: "Development Tools",
      items: [
        {
          icon: <GitBranch className="text-orange-500" size={32} />,
          name: "Version Control",
          description: "Code management and collaboration tools",
          features: ["Git Workflow", "Code Review", "Branch Management", "CI/CD"],
          color: "orange"
        },
        {
          icon: <Zap className="text-yellow-500" size={32} />,
          name: "Development Tools",
          description: "Tools for efficient development workflow",
          features: ["Hot Reloading", "Debug Tools", "Testing Suite", "Code Quality"],
          color: "yellow"
        }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 py-12 relative overflow-hidden">
      {/* 3D Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Floating 3D Tech Icons */}
        <div 
          className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-blue-400/20 to-cyan-400/20 rounded-full blur-xl animate-float-3d transform-gpu"
          style={{
            transform: `translateZ(0) rotateX(45deg) rotateY(${mousePosition.x * 0.1}deg)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
        <div 
          className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full blur-lg animate-float-3d-delayed transform-gpu"
          style={{
            transform: `translateZ(0) rotateX(-30deg) rotateY(${mousePosition.y * 0.1}deg)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
        <div 
          className="absolute bottom-32 left-1/4 w-40 h-40 bg-gradient-to-br from-green-400/15 to-teal-400/15 rounded-full blur-2xl animate-pulse-3d transform-gpu"
          style={{
            transform: `translateZ(0) rotateX(60deg) rotateY(-${mousePosition.x * 0.05}deg)`,
            transition: 'transform 0.3s ease-out'
          }}
        />
        
        {/* 3D Geometric Tech Shapes */}
        <div className="absolute top-1/3 right-1/4 w-16 h-16 bg-gradient-to-br from-orange-400/30 to-red-400/30 transform rotate-45 animate-spin-3d blur-sm" />
        <div className="absolute bottom-1/4 right-1/3 w-12 h-12 bg-gradient-to-br from-indigo-400/30 to-purple-400/30 transform rotate-12 animate-bounce-3d blur-sm" />
        
        {/* Floating Code Symbols */}
        <div className="absolute top-1/4 left-1/3 animate-sparkle-3d">
          <Code2 className="text-blue-400/40 w-8 h-8 transform-gpu" style={{ transform: 'rotateZ(45deg)' }} />
        </div>
        <div className="absolute top-2/3 right-1/2 animate-sparkle-3d-delayed">
          <Database className="text-green-400/40 w-6 h-6 transform-gpu" style={{ transform: 'rotateZ(-30deg)' }} />
        </div>
        <div className="absolute bottom-1/3 left-1/2 animate-float-3d">
          <Server className="text-purple-400/40 w-7 h-7 transform-gpu" style={{ transform: 'rotateX(45deg)' }} />
        </div>
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Hero Section with 3D Effects */}
        <div className="text-center mb-16 animate-slide-up-3d">
          <div className="inline-flex items-center px-6 py-3 bg-white/80 backdrop-blur-md rounded-full shadow-lg border border-white/20 mb-6 transform hover:scale-105 transition-all duration-300 hover:shadow-xl">
            <Cpu className="text-purple-600 mr-3 animate-pulse-3d" size={24} />
            <span className="text-purple-700 font-medium">Technology Stack</span>
          </div>
          
          <h1 className="text-5xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent animate-text-shimmer">
            Our Technology Stack
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto animate-fade-in-up-3d">
            Discover the cutting-edge technologies powering SnipX's video processing capabilities
          </p>
        </div>

        {/* Technology Stacks with Enhanced 3D Animations */}
        <div className="space-y-20">
          {techStacks.map((stack, stackIndex) => (
            <div 
              key={stackIndex}
              data-index={stackIndex}
              className={`tech-section transition-all duration-1000 ${
                visibleSections.includes(stackIndex) 
                  ? 'opacity-100 translate-y-0' 
                  : 'opacity-0 translate-y-8'
              }`}
            >
              <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center animate-slide-in-3d">
                <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {stack.category}
                </span>
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {stack.items.map((tech, techIndex) => (
                  <div 
                    key={techIndex} 
                    className={`group bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-white/20 p-8 transition-all duration-500 transform hover:-translate-y-4 hover:scale-105 hover:shadow-2xl animate-card-float-3d interactive-3d`}
                    style={{ 
                      animationDelay: `${(stackIndex * 2 + techIndex) * 200}ms`,
                      transform: visibleSections.includes(stackIndex) 
                        ? 'translateY(0) translateZ(0) rotateX(0deg)' 
                        : 'translateY(30px) translateZ(-20px) rotateX(-10deg)'
                    }}
                  >
                    <div className="flex items-start gap-6">
                      {/* 3D Icon Container */}
                      <div className={`flex-shrink-0 relative group-hover:scale-110 transition-all duration-300`}>
                        <div className={`p-4 rounded-2xl bg-gradient-to-br from-${tech.color}-50 to-${tech.color}-100 group-hover:from-${tech.color}-100 group-hover:to-${tech.color}-200 shadow-lg transform group-hover:rotateY(15deg) group-hover:rotateX(10deg) transition-all duration-300`}>
                          <div className="transform group-hover:scale-110 group-hover:rotateZ(5deg) transition-all duration-300">
                            {tech.icon}
                          </div>
                          {/* Glow Effect */}
                          <div className={`absolute inset-0 bg-${tech.color}-400/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 transform scale-150`}></div>
                        </div>
                        
                        {/* Floating Sparkles */}
                        <div className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-all duration-300 animate-sparkle-3d">
                          <Sparkles className={`w-4 h-4 text-${tech.color}-400`} />
                        </div>
                      </div>

                      <div className="flex-1">
                        <h3 className={`text-2xl font-semibold text-gray-900 mb-3 group-hover:text-${tech.color}-700 transition-colors duration-300`}>
                          {tech.name}
                        </h3>
                        <p className="text-gray-600 mb-6 leading-relaxed group-hover:text-gray-700 transition-colors duration-300">
                          {tech.description}
                        </p>
                        
                        {/* 3D Feature Tags */}
                        <div className="grid grid-cols-2 gap-3">
                          {tech.features.map((feature, featureIndex) => (
                            <div 
                              key={featureIndex}
                              className={`flex items-center text-sm text-gray-600 bg-gradient-to-r from-${tech.color}-50 to-${tech.color}-100 rounded-xl px-4 py-2 shadow-sm transform hover:scale-105 hover:-translate-y-1 transition-all duration-300 hover:shadow-md animate-bounce-in-3d`}
                              style={{ animationDelay: `${featureIndex * 100}ms` }}
                            >
                              <span className={`w-2 h-2 bg-${tech.color}-500 rounded-full mr-3 animate-pulse`}></span>
                              <span className="font-medium">{feature}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* 3D Hover Overlay */}
                    <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-transparent via-transparent to-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    
                    {/* Interactive Border Glow */}
                    <div className={`absolute inset-0 rounded-2xl border-2 border-${tech.color}-300/0 group-hover:border-${tech.color}-300/50 transition-all duration-300`}></div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Call to Action with 3D Effects */}
        <div className="mt-20 text-center animate-slide-up-3d">
          <div className="bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-2xl p-8 shadow-2xl transform hover:scale-105 transition-all duration-300 hover:shadow-3xl">
            <h3 className="text-2xl font-bold text-white mb-4 animate-text-reveal">
              Ready to Experience the Power?
            </h3>
            <p className="text-purple-100 mb-6 animate-fade-in-up-3d">
              Try SnipX today and see how our advanced technology stack can transform your video editing workflow.
            </p>
            <button className="bg-white text-purple-600 px-8 py-3 rounded-xl font-semibold hover:bg-gray-100 transition-all duration-300 transform hover:scale-105 hover:shadow-lg btn-3d">
              Get Started Now
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Technologies;