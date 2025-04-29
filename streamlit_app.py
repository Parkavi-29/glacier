// ------------------------------
// 1. Define AOI (Gangotri Glacier)
// ------------------------------
var aoi = ee.Geometry.Rectangle([79.03, 30.94, 79.10, 31.02]);
Map.addLayer(ee.FeatureCollection(ee.Feature(aoi)), {color: 'black'}, '📏 AOI Boundary');

// ------------------------------
// 2. Year Ranges
// ------------------------------
var yearsPre = ee.List.sequence(2001, 2012);
var yearsPost = ee.List.sequence(2013, 2023);

// ------------------------------
// 3. Load Landsat Collections
// ------------------------------
var l5_7 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
              .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"));
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");

// ------------------------------
// 4. Glacier Processing (NDSI only)
// ------------------------------
function getGlacier(year, collection, greenBand, swirBand) {
  var start = ee.Date.fromYMD(year, 6, 1);
  var end = ee.Date.fromYMD(year, 9, 30);
  
  var composite = collection
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUD_COVER', 20))
    .map(function(img) {
      var mask = img.select('QA_PIXEL').bitwiseAnd(1 << 3).eq(0);
      return img.updateMask(mask);
    })
    .median();
  
  var bandNames = composite.bandNames();
  var green = ee.Image(
    ee.Algorithms.If(
      bandNames.contains('SR_B2') || bandNames.contains('SR_B3'),
      composite.select(greenBand),
      ee.Image(0).rename(greenBand)
    )
  );
  var swir = ee.Image(
    ee.Algorithms.If(
      bandNames.contains('SR_B5') || bandNames.contains('SR_B6'),
      composite.select(swirBand),
      ee.Image(0).rename(swirBand)
    )
  );
  
  var ndsi = green.subtract(swir).divide(green.add(swir)).rename('NDSI');
  var glacier = ndsi.gt(0.4).selfMask().set('year', year);
  
  // Return only if glacier is not empty
  var area = glacier.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: aoi,
    scale: 30,
    maxPixels: 1e13
  }).get('NDSI');
  
  return ee.Algorithms.If(ee.Number(area).gt(0), glacier, null);
}

// ------------------------------
// 5. Generate Glacier Mask Images
// ------------------------------
var glaciersList = [];

yearsPre.getInfo().forEach(function(year) {
  var glacier = getGlacier(year, l5_7, 'SR_B2', 'SR_B5');
  if (glacier !== null) {
    glaciersList.push(glacier);
  }
});

yearsPost.getInfo().forEach(function(year) {
  var glacier = getGlacier(year, l8, 'SR_B3', 'SR_B6');
  if (glacier !== null) {
    glaciersList.push(glacier);
  }
});

// ------------------------------
// 6. Build an ImageCollection
// ------------------------------
var glaciersCollection = ee.ImageCollection.fromImages(glaciersList);

// ------------------------------
// 7. Visualization
// ------------------------------
var visParams = {
  palette: ['#00FFFF'],
  min: 0,
  max: 1,
  opacity: 0.8
};

// Years list (filtered only available years)
var yearsList = glaciersCollection.aggregate_array('year');

print('✅ Available glacier years:', yearsList);

// ------------------------------
// 8. Time Slider Animation
// ------------------------------
var slider = ui.DateSlider({
  start: '2001-01-01',
  end: '2023-12-31',
  period: 365,
  value: '2001-01-01',
  onChange: function(dateRange) {
    Map.layers().reset();
    Map.addLayer(ee.FeatureCollection(ee.Feature(aoi)), {color: 'black'}, '📏 AOI Boundary');

    var selectedYear = ee.Date(dateRange.start()).get('year');
    var image = glaciersCollection.filter(ee.Filter.eq('year', selectedYear)).first();
    
    if (image) {
      Map.addLayer(image.visualize(visParams), {}, '🧊 Glacier ' + selectedYear.getInfo());
    } else {
      print('⚠️ No glacier data available for year:', selectedYear);
    }
  }
});

Map.add(slider);

// ------------------------------
// 9. Final Center and Basemap
// ------------------------------
Map.setOptions('SATELLITE');
Map.centerObject(aoi, 12);
