# Additional Missing Components

## 4. Quality Flag Validator and Other Validators

class QualityFlagValidator:
    """Validate ARGO quality flags"""
    
    def __init__(self):
        self.quality_meanings = {
            1: 'good_data',
            2: 'probably_good_data', 
            3: 'probably_bad_data_correctable',
            4: 'bad_data',
            5: 'value_changed',
            8: 'estimated_value',
            9: 'missing_value'
        }
        
    def validate(self, data: Dict) -> Dict:
        """Validate data based on quality flags"""
        quality_flags = data.get('quality_flags', [])
        measurements = data.get('measurements', {})
        
        validated_data = data.copy()
        
        for param in ['temperature', 'salinity']:
            if param in measurements and quality_flags:
                param_data = measurements[param]
                filtered_data = []
                
                for i, (value, flag) in enumerate(zip(param_data, quality_flags)):
                    if flag in [1, 2]:  # Good or probably good
                        filtered_data.append(value)
                    else:
                        filtered_data.append(None)  # Mark as invalid
                
                validated_data['measurements'][param] = filtered_data
        
        return validated_data

class GeospatialBoundsValidator:
    """Validate geographic coordinates"""
    
    def __init__(self):
        self.valid_bounds = {
            'latitude': (-90, 90),
            'longitude': (-180, 180)
        }
        
        # Ocean bounds (simplified)
        self.ocean_bounds = {
            'indian_ocean': {
                'lat': (-60, 30),
                'lon': (20, 147)
            }
        }
    
    def validate(self, data: Dict) -> bool:
        """Check if coordinates are valid"""
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if lat is None or lon is None:
            return False
        
        # Check basic bounds
        if not (self.valid_bounds['latitude'][0] <= lat <= self.valid_bounds['latitude'][1]):
            return False
        if not (self.valid_bounds['longitude'][0] <= lon <= self.valid_bounds['longitude'][1]):
            return False
        
        # Check if in ocean (simplified check)
        return self.is_in_ocean(lat, lon)
    
    def is_in_ocean(self, lat: float, lon: float) -> bool:
        """Simple ocean check"""
        # This is very simplified - use proper ocean mask in production
        for ocean, bounds in self.ocean_bounds.items():
            if (bounds['lat'][0] <= lat <= bounds['lat'][1] and
                bounds['lon'][0] <= lon <= bounds['lon'][1]):
                return True
        return True  # Assume ocean for now

class TemporalConsistencyValidator:
    """Validate temporal consistency"""
    
    def __init__(self):
        self.argo_start_date = datetime(1999, 1, 1)
        self.max_future_days = 7  # Allow some future dates for processing delays
    
    def validate(self, data: Dict) -> bool:
        """Check if date is reasonable"""
        profile_date = data.get('profile_date')
        
        if not profile_date:
            return False
        
        if isinstance(profile_date, str):
            try:
                profile_date = datetime.fromisoformat(profile_date)
            except:
                return False
        
        # Check if date is after ARGO start
        if profile_date < self.argo_start_date:
            return False
        
        # Check if date is not too far in future
        max_future_date = datetime.now() + timedelta(days=self.max_future_days)
        if profile_date > max_future_date:
            return False
        
        return True

## 5. Missing Extractor Classes

class MetadataExtractor:
    """Extract metadata from ARGO profiles"""
    
    def extract(self, netcdf_data) -> Dict:
        """Extract profile metadata"""
        metadata = {
            'platform_number': str(netcdf_data.attrs.get('platform_number', 'unknown')),
            'cycle_number': int(netcdf_data.attrs.get('cycle_number', 0)),
            'institution': str(netcdf_data.attrs.get('institution', 'unknown')),
            'data_center': str(netcdf_data.attrs.get('data_centre', 'unknown')),
            'wmo_inst_type': str(netcdf_data.attrs.get('wmo_inst_type', 'unknown')),
            'positioning_system': str(netcdf_data.attrs.get('positioning_system', 'GPS')),
            'profile_direction': str(netcdf_data.attrs.get('direction', 'A')),  # A=ascending, D=descending
        }
        
        # Extract float configuration
        if 'CONFIG_MISSION_NUMBER' in netcdf_data.variables:
            metadata['mission_number'] = int(netcdf_data['CONFIG_MISSION_NUMBER'].values[0])
        
        return metadata

class ProfileSummaryExtractor:
    """Extract profile summaries for vector storage"""
    
    def extract(self, profile_data: Dict) -> Dict:
        """Create searchable profile summary"""
        measurements = profile_data.get('measurements', {})
        
        summary = {
            'profile_id': profile_data.get('profile_id'),
            'location_text': f"Lat: {profile_data.get('latitude', 0):.2f}, Lon: {profile_data.get('longitude', 0):.2f}",
            'date_text': profile_data.get('profile_date', '').strftime('%Y-%m-%d') if profile_data.get('profile_date') else '',
            'depth_range': f"0 to {max(measurements.get('pressure', [0]))}m" if measurements.get('pressure') else "Surface only",
            'parameters': ', '.join([k for k in measurements.keys() if measurements[k]]),
        }
        
        # Temperature summary
        if measurements.get('temperature'):
            temps = [t for t in measurements['temperature'] if t is not None]
            if temps:
                summary['temperature_range'] = f"{min(temps):.1f} to {max(temps):.1f}°C"
                summary['surface_temperature'] = f"{temps[0]:.1f}°C"
        
        # Salinity summary
        if measurements.get('salinity'):
            sals = [s for s in measurements['salinity'] if s is not None]
            if sals:
                summary['salinity_range'] = f"{min(sals):.2f} to {max(sals):.2f} PSU"
        
        # Create searchable text
        summary['searchable_text'] = f"""
        ARGO Profile {summary['profile_id']} from {summary['location_text']} 
        on {summary['date_text']}. Measurements: {summary['parameters']}.
        Depth: {summary['depth_range']}. Temperature: {summary.get('temperature_range', 'N/A')}.
        Salinity: {summary.get('salinity_range', 'N/A')}.
        """
        
        return summary

class AnomalyDetector:
    """Detect anomalies in oceanographic profiles"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    def detect_anomalies(self, profile_data: Dict) -> List[Dict]:
        """Detect various types of anomalies"""
        anomalies = []
        measurements = profile_data.get('measurements', {})
        
        # Parameter-based anomalies
        for param in ['temperature', 'salinity']:
            if param in measurements and measurements[param]:
                param_anomalies = self._detect_parameter_anomalies(
                    measurements[param], param, profile_data
                )
                anomalies.extend(param_anomalies)
        
        # Profile shape anomalies
        profile_anomalies = self._detect_profile_anomalies(measurements, profile_data)
        anomalies.extend(profile_anomalies)
        
        return anomalies
    
    def _detect_parameter_anomalies(self, values: List[float], parameter: str, profile_info: Dict) -> List[Dict]:
        """Detect anomalies in individual parameters"""
        anomalies = []
        clean_values = [v for v in values if v is not None]
        
        if len(clean_values) < 3:
            return anomalies
        
        # Statistical outliers using IQR
        Q1 = np.percentile(clean_values, 25)
        Q3 = np.percentile(clean_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        for i, value in enumerate(values):
            if value is not None and (value < lower_bound or value > upper_bound):
                anomalies.append({
                    'type': 'statistical_outlier',
                    'parameter': parameter,
                    'value': value,
                    'depth_index': i,
                    'severity': 'high' if abs(value - np.mean(clean_values)) > 3 * np.std(clean_values) else 'medium',
                    'profile_id': profile_info.get('profile_id')
                })
        
        return anomalies
    
    def _detect_profile_anomalies(self, measurements: Dict, profile_info: Dict) -> List[Dict]:
        """Detect anomalies in profile shape/structure"""
        anomalies = []
        
        # Check for temperature inversions (unusual warming with depth)
        if 'temperature' in measurements and 'pressure' in measurements:
            temps = measurements['temperature']
            pressures = measurements['pressure']
            
            for i in range(1, len(temps)):
                if temps[i] is not None and temps[i-1] is not None:
                    if temps[i] > temps[i-1] + 2.0 and pressures[i] > pressures[i-1]:  # Warming with depth
                        anomalies.append({
                            'type': 'temperature_inversion',
                            'depth_index': i,
                            'temperature_increase': temps[i] - temps[i-1],
                            'severity': 'medium',
                            'profile_id': profile_info.get('profile_id')
                        })
        
        return anomalies

## 6. Intent Classification Components

class IntentClassifier:
    """Classify user intent for oceanographic queries"""
    
    def __init__(self, intents: List[str]):
        self.intents = intents
        self.intent_patterns = {
            'profile_query': [
                'show', 'display', 'plot', 'profile', 'depth', 'vertical',
                'temperature', 'salinity', 'oxygen'
            ],
            'comparison': [
                'compare', 'difference', 'vs', 'versus', 'between',
                'contrast', 'different'
            ],
            'trajectory': [
                'trajectory', 'path', 'movement', 'track', 'float',
                'journey', 'route'
            ],
            'statistical_analysis': [
                'average', 'mean', 'median', 'statistics', 'correlation',
                'trend', 'analysis', 'distribution'
            ],
            'anomaly_detection': [
                'unusual', 'anomaly', 'strange', 'abnormal', 'outlier',
                'extreme', 'rare'
            ]
        }
    
    def predict(self, query: str) -> str:
        """Predict intent from query text"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        # Return intent with highest score, default to profile_query
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        return 'profile_query'

## 7. Geospatial and Temporal NER Components

class GeospatialNER:
    """Extract geographic entities from text"""
    
    def __init__(self):
        self.regions = {
            'arabian sea': {'lat': (8, 25), 'lon': (50, 77)},
            'bay of bengal': {'lat': (5, 22), 'lon': (77, 95)},
            'indian ocean': {'lat': (-60, 30), 'lon': (20, 147)},
            'equator': {'lat': (-2, 2), 'lon': (-180, 180)},
            'equatorial': {'lat': (-5, 5), 'lon': (-180, 180)},
            'tropical': {'lat': (-23.5, 23.5), 'lon': (-180, 180)},
            'northern indian ocean': {'lat': (0, 30), 'lon': (30, 100)},
            'southern indian ocean': {'lat': (-60, 0), 'lon': (30, 147)}
        }
        
        # Coordinate patterns
        self.coord_patterns = [
            r'(\d+\.?\d*)[°\s]*([NS]?)\s*[,\s]\s*(\d+\.?\d*)[°\s]*([EW]?)',
            r'lat[itude]*:\s*(-?\d+\.?\d*)',
            r'lon[gitude]*:\s*(-?\d+\.?\d*)'
        ]
    
    def extract(self, text: str) -> List[Dict]:
        """Extract geographic entities"""
        entities = []
        text_lower = text.lower()
        
        # Extract named regions
        for region, bounds in self.regions.items():
            if region in text_lower:
                entities.append({
                    'type': 'region',
                    'name': region,
                    'bounds': bounds,
                    'confidence': 0.9
                })
        
        # Extract coordinates using regex
        import re
        for pattern in self.coord_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    'type': 'coordinates',
                    'raw_match': match,
                    'confidence': 0.8
                })
        
        return entities

class TemporalNER:
    """Extract temporal entities from text"""
    
    def __init__(self):
        self.months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        self.seasons = {
            'monsoon': [6, 7, 8, 9],
            'winter': [12, 1, 2],
            'summer': [6, 7, 8],
            'spring': [3, 4, 5],
            'pre-monsoon': [4, 5],
            'post-monsoon': [10, 11]
        }
    
    def extract(self, text: str) -> List[Dict]:
        """Extract temporal entities"""
        entities = []
        text_lower = text.lower()
        
        # Extract months
        for month_name, month_num in self.months.items():
            if month_name in text_lower:
                entities.append({
                    'type': 'month',
                    'name': month_name,
                    'number': month_num,
                    'confidence': 0.9
                })
        
        # Extract seasons
        for season_name, months in self.seasons.items():
            if season_name in text_lower:
                entities.append({
                    'type': 'season',
                    'name': season_name,
                    'months': months,
                    'confidence': 0.9
                })
        
        # Extract years using regex
        import re
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        for year in year_matches:
            entities.append({
                'type': 'year',
                'value': int(year),
                'confidence': 0.95
            })
        
        # Extract relative time
        relative_patterns = {
            'last month': {'type': 'relative', 'offset': -1, 'unit': 'month'},
            'past year': {'type': 'relative', 'offset': -1, 'unit': 'year'},
            'recent': {'type': 'relative', 'offset': -3, 'unit': 'month'},
            'latest': {'type': 'relative', 'offset': -1, 'unit': 'month'}
        }
        
        for pattern, info in relative_patterns.items():
            if pattern in text_lower:
                entities.append({
                    'type': 'relative_time',
                    'pattern': pattern,
                    'offset': info['offset'],
                    'unit': info['unit'],
                    'confidence': 0.8
                })
        
        return entities

class OceanParameterNER:
    """Extract oceanographic parameter entities"""
    
    def __init__(self):
        self.parameters = {
            'temperature': ['temperature', 'temp', 'thermal', '°c', 'celsius'],
            'salinity': ['salinity', 'sal', 'salt', 'psu', 'practical salinity'],
            'oxygen': ['oxygen', 'o2', 'dissolved oxygen', 'do'],
            'pressure': ['pressure', 'depth', 'pres', 'dbar'],
            'chlorophyll': ['chlorophyll', 'chl', 'chla', 'chlorophyll-a'],
            'nitrate': ['nitrate', 'no3', 'nitrogen', 'nutrients'],
            'ph': ['ph', 'acidity', 'alkalinity'],
            'density': ['density', 'sigma', 'potential density']
        }
        
        self.derived_parameters = {
            'thermocline': ['thermocline', 'thermal gradient'],
            'halocline': ['halocline', 'salinity gradient'],
            'mixed_layer': ['mixed layer', 'surface layer', 'mld'],
            'deep_water': ['deep water', 'bottom water', 'abyssal']
        }
    
    def extract(self, text: str) -> List[Dict]:
        """Extract parameter entities"""
        entities = []
        text_lower = text.lower()
        
        # Extract standard parameters
        for param_name, aliases in self.parameters.items():
            for alias in aliases:
                if alias in text_lower:
                    entities.append({
                        'type': 'parameter',
                        'name': param_name,
                        'alias': alias,
                        'confidence': 0.9
                    })
                    break  # Only add once per parameter
        
        # Extract derived parameters
        for param_name, aliases in self.derived_parameters.items():
            for alias in aliases:
                if alias in text_lower:
                    entities.append({
                        'type': 'derived_parameter',
                        'name': param_name,
                        'alias': alias,
                        'confidence': 0.8
                    })
                    break
        
        return entities

## 8. Region Bounds Helper

def get_region_bounds(region_name: str) -> Dict:
    """Get geographic bounds for named regions"""
    region_bounds = {
        'arabian sea': {'west': 50, 'east': 77, 'south': 8, 'north': 25},
        'bay of bengal': {'west': 77, 'east': 95, 'south': 5, 'north': 22},
        'indian ocean': {'west': 20, 'east': 147, 'south': -60, 'north': 30},
        'equatorial indian ocean': {'west': 40, 'east': 100, 'south': -10, 'north': 10},
        'tropical indian ocean': {'west': 30, 'east': 120, 'south': -23.5, 'north': 23.5},
        'southern indian ocean': {'west': 20, 'east': 147, 'south': -60, 'north': 0},
        'northern indian ocean': {'west': 30, 'east': 100, 'south': 0, 'north': 30}
    }
    
    return region_bounds.get(region_name.lower(), {
        'west': -180, 'east': 180, 'south': -90, 'north': 90
    })

## 9. Coordinate Parser

def parse_coordinates(coord_string: str) -> Tuple[float, float]:
    """Parse coordinate string to lat, lon"""
    import re
    
    # Try different formats
    patterns = [
        r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)',  # "15.5, 68.2"
        r'(\d+\.?\d*)[°\s]*([NS])[,\s]*(\d+\.?\d*)[°\s]*([EW])',  # "15.5°N, 68.2°E"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, coord_string)
        if match:
            if len(match.groups()) == 2:
                return float(match.group(1)), float(match.group(2))
            elif len(match.groups()) == 4:
                lat = float(match.group(1))
                if match.group(2).upper() == 'S':
                    lat = -lat
                lon = float(match.group(3))
                if match.group(4).upper() == 'W':
                    lon = -lon
                return lat, lon
    
    # Default fallback
    return 0.0, 0.0

## 10. Profile Popup Creator for Maps

def create_profile_popup(profile_data: Dict) -> str:
    """Create HTML popup content for map markers"""
    measurements = profile_data.get('measurements', {})
    
    html = f"""
    <div style='width: 200px'>
        <h4>ARGO Profile {profile_data.get('profile_id', 'Unknown')}</h4>
        <p><b>Float ID:</b> {profile_data.get('float_id', 'Unknown')}</p>
        <p><b>Date:</b> {profile_data.get('profile_date', 'Unknown')}</p>
        <p><b>Location:</b> {profile_data.get('latitude', 0):.2f}°, {profile_data.get('longitude', 0):.2f}°</p>
    """
    
    if measurements.get('temperature'):
        temps = [t for t in measurements['temperature'] if t is not None]
        if temps:
            html += f"<p><b>Temperature:</b> {min(temps):.1f} - {max(temps):.1f}°C</p>"
    
    if measurements.get('salinity'):
        sals = [s for s in measurements['salinity'] if s is not None]
        if sals:
            html += f"<p><b>Salinity:</b> {min(sals):.2f} - {max(sals):.2f} PSU</p>"
    
    html += "</div>"
    return html

## 11. Multi-Store Writer

class MultiStoreWriter:
    """Write data to multiple storage backends"""
    
    def __init__(self, db_manager, vector_store):
        self.db_manager = db_manager
        self.vector_store = vector_store
    
    async def write_profile(self, profile_data: Dict):
        """Write profile to all storage backends"""
        # Write to relational database
        await self.db_manager.store_data([profile_data])
        
        # Create embeddings and store in vector database
        summary_extractor = ProfileSummaryExtractor()
        summary = summary_extractor.extract(profile_data)
        
        # Add to vector store (simplified)
        self.vector_store.add_texts(
            texts=[summary['searchable_text']],
            metadatas=[summary],
            ids=[profile_data['profile_id']]
        )