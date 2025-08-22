# Jekyll plugin to read lecture meta.yml files and make them available in Liquid templates
require 'yaml'

module LectureReader
  class Generator < Jekyll::Generator
    safe true
    priority :high

    def generate(site)
      # Path to lectures directory (relative to website root)
      lectures_path = File.join(site.source, '..', 'lectures')
      
      # Initialize lectures array
      lectures = []
      
      # Read all lecture directories
      if Dir.exist?(lectures_path)
        Dir.glob(File.join(lectures_path, 'lecture_*')).sort.each do |lecture_dir|
          lecture_data = read_lecture_meta(lecture_dir, site)
          # Only include visible lectures
          if lecture_data && (lecture_data['visible'].nil? || lecture_data['visible'])
            lectures << lecture_data
          end
        end
      end
      
      # Sort lectures by number
      lectures.sort_by! { |lecture| lecture['lecture_number'].to_i }
      
      # Make available to all pages
      # If running in an environment without ../lectures (e.g., GitHub Pages project repo),
      # do NOT overwrite existing data loaded from _data/lectures.yml
      if !lectures.empty?
        site.data['lectures'] = lectures
      end
      
      # Also create individual lecture pages if needed
      create_lecture_pages(site, lectures) if site.config['create_lecture_pages']
    end

    private

    def read_lecture_meta(lecture_dir, site)
      # Look for meta.yml in different locations
      meta_paths = [
        File.join(lecture_dir, 'meta.yml'),
        File.join(lecture_dir, 'materials', 'meta.yml')
      ]
      
      meta_file = meta_paths.find { |path| File.exist?(path) }
      return nil unless meta_file
      
      begin
        # Read and parse YAML with safe loading to handle dates
        meta_content = File.read(meta_file)
        meta_data = YAML.safe_load(meta_content, permitted_classes: [Date, Time])
        
        # Skip if YAML is empty or not a hash
        return nil unless meta_data.is_a?(Hash)
        
        # Add computed fields
        lecture_slug = File.basename(lecture_dir)
        relative_path = File.join('..', 'lectures', lecture_slug)
        
        meta_data['slug'] = lecture_slug
        meta_data['directory'] = lecture_dir
        meta_data['relative_path'] = relative_path
        
        # Convert date to "Aug 26" format
        if meta_data['date'].is_a?(Date)
          meta_data['date'] = meta_data['date'].strftime('%b %d')
        elsif meta_data['date'].is_a?(Time)
          meta_data['date'] = meta_data['date'].strftime('%b %d')
        elsif meta_data['date'].is_a?(String) && meta_data['date'].match(/^\d{4}-\d{2}-\d{2}$/)
          # Parse string date and format
          begin
            parsed_date = Date.parse(meta_data['date'])
            meta_data['date'] = parsed_date.strftime('%b %d')
          rescue => e
            # Keep original format if parsing fails
            Jekyll.logger.warn "LectureReader:", "Could not parse date #{meta_data['date']}: #{e.message}"
          end
        end
        
        # Process material URLs to be relative to the lecture directory
        process_material_urls(meta_data, relative_path)
        
        # Set default values if missing
        meta_data['lecture_number'] ||= extract_lecture_number(lecture_slug)
        meta_data['type'] ||= 'lecture'
        meta_data['difficulty'] ||= 'intermediate'
        meta_data['phase'] ||= 'foundations'
        
        return meta_data
        
      rescue => e
        Jekyll.logger.warn "LectureReader:", "Error reading #{meta_file}: #{e.message}"
        return nil
      end
    end

    def process_material_urls(meta_data, base_path)
      # Handle materials section if it exists
      # Note: We no longer modify URLs here - they're handled by the sync system
      # and the schedule page will add the /materials/ prefix as needed
      
      # Handle legacy links section (convert to materials format)
      if meta_data['links'].is_a?(Array)
        materials = {}
        
        meta_data['links'].each do |section|
          next unless section.is_a?(Hash) && section['section'] && section['items']
          
          section_name = section['section'].downcase.gsub(/\s+/, '_')
          materials[section_name] = section['items'].map do |item|
            next unless item.is_a?(Hash)
            
            # Convert legacy format to new materials format
            material = {
              'name' => item['name'],
              'url' => item['url'],
              'type' => 'resource'
            }
            
            # URLs are now handled by the sync system and schedule page
            # No need to modify them here
            
            material
          end.compact
        end
        
        meta_data['materials'] = materials unless materials.empty?
      end
    end

    def extract_lecture_number(slug)
      match = slug.match(/lecture_(\d+)/)
      match ? match[1].to_i : 0
    end

    def create_lecture_pages(site, lectures)
      lectures.each do |lecture|
        # Create individual pages for each lecture if needed
        page = LecturePage.new(site, lecture)
        site.pages << page
      end
    end
  end

  # Individual lecture page class
  class LecturePage < Jekyll::Page
    def initialize(site, lecture_data)
      @site = site
      @base = site.source
      @dir = "lectures"
      @name = "#{lecture_data['slug']}.html"

      self.process(@name)
      self.read_yaml(File.join(@base, '_layouts'), 'lecture.html')
      
      self.data['lecture'] = lecture_data
      self.data['title'] = lecture_data['title']
    end
  end
end

# Hook for liquid filters
module LectureFilters
  def format_duration(input)
    return "" unless input
    
    minutes = input.to_i
    if minutes >= 60
      hours = minutes / 60
      remaining_minutes = minutes % 60
      remaining_minutes > 0 ? "#{hours}h #{remaining_minutes}m" : "#{hours}h"
    else
      "#{minutes}m"
    end
  end

  def lecture_type_class(type)
    case type.to_s.downcase
    when 'lecture'
      'lecture-number'
    when 'lab', 'case_study'
      'lecture-number assignment-number'
    when 'exam', 'midterm', 'final'
      'lecture-number exam-number'
    when 'hackathon'
      'lecture-number hackathon-number'
    else
      'lecture-number'
    end
  end

  def lecture_materials_by_type(materials, type)
    return [] unless materials && materials[type]
    materials[type]
  end

  def format_assignment_info(assignments)
    info = []
    
    if assignments
      if assignments['due_today']
        assignments['due_today'].each do |assignment|
          info << "#{assignment['name']} Due"
        end
      end
      
      if assignments['released_today']
        assignments['released_today'].each do |assignment|
          info << "#{assignment['name']} Released"
        end
      end
    end
    
    info.join(", ")
  end
end

Liquid::Template.register_filter(LectureFilters)