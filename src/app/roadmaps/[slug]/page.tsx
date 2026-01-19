import React from 'react';
import { roadmaps } from '@/data/roadmaps';
import Link from 'next/link';

export async function generateStaticParams() {
  const slugs = Object.keys(roadmaps).map((slug) => ({
    slug,
  }));
  return slugs;
}

type RoadmapPageProps = {
  params: {
    slug: string;
  };
};

const RoadmapPage = ({ params }: RoadmapPageProps) => {
  const { slug } = params;
  const roadmap = roadmaps[slug];

  if (!roadmap) {
    return <div className="py-16 text-center">Roadmap not found.</div>;
  }

  return (
    <div className="py-16">
      <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">{roadmap.title}</h1>
      <p className="text-xl text-gray-400 mb-8">{roadmap.description}</p>

      <div className="space-y-8">
        {roadmap.steps.map((step, index) => (
          <div key={index} className="bg-gray-800 p-6 rounded-lg">
            <h2 className="text-3xl font-bold mb-2 text-gray-100">{step.title}</h2>
            <p className="text-gray-400 mb-4">{step.description}</p>
            <ul className="space-y-2">
              {step.resources.map((resource, rIndex) => (
                <li key={rIndex}>
                  <Link href={resource.url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                    {resource.title}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RoadmapPage;
