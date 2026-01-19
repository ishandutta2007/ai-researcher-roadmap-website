import React from 'react';
import Link from 'next/link';

const HomePage = () => {
  const roadmaps = [
    { title: 'Frontend Developer', slug: 'frontend', description: 'Step by step guide to becoming a modern frontend developer in 2024' },
    { title: 'Backend Developer', slug: 'backend', description: 'Step by step guide to becoming a modern backend developer in 2024' },
    { title: 'DevOps', slug: 'devops', description: 'Step by step guide for DevOps, SRE or any other operations role in 2024' },
    { title: 'Full Stack Developer', slug: 'full-stack', description: 'Step by step guide to becoming a full stack developer in 2024' },
    { title: 'Android Developer', slug: 'android', description: 'Step by step guide to becoming an Android developer in 2024' },
    { title: 'PostgreSQL DBA', slug: 'postgresql-dba', description: 'Step by step guide to becoming a PostgreSQL DBA in 2024' },
  ];

  return (
    <div>
      <div className="text-center py-16">
        <h1 className="text-5xl font-bold mb-4">AI Researcher Roadmap</h1>
        <p className="text-xl text-gray-400 mb-8">
          Community driven roadmaps, guides and articles for developers.
        </p>
        <div className="max-w-2xl mx-auto">
          <input
            type="text"
            placeholder="Search for a roadmap..."
            className="w-full p-4 rounded-full bg-gray-800 text-white border-2 border-gray-700 focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-16">
        {roadmaps.map((roadmap) => (
          <Link key={roadmap.slug} href={`/roadmaps/${roadmap.slug}`}>
            <div className="bg-gray-800 p-6 rounded-lg text-center h-full hover:bg-gray-700 transition-colors">
              <h2 className="text-2xl font-bold mb-2">{roadmap.title}</h2>
              <p className="text-gray-400">{roadmap.description}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default HomePage;
