use std::path::PathBuf;

use minifb::{Key, Window, WindowOptions};

use super::draw::{self, DrawOption, ToBuffer};
use crate::coco::object_detection::{Annotation, BTreemapDataset};
use crate::utils;

/// Visualize the annotations for the given image id.
///
/// # Errors
///
/// Will return `Err` if `img_id` is not present in the dataset.
pub fn img_anns(dataset: &BTreemapDataset, img_id: u64, draw_option: DrawOption) -> Result<(), Box<dyn std::error::Error>> {
    let anns = dataset.get_img_anns(img_id)?;
    let img_name = &dataset.get_img(img_id)?.file_name;
    let img_path = dataset.image_folder.join(img_name);

    self::anns(&img_path, &anns, draw_option)?;

    Ok(())
}

/// Display the given image in a window.
///
/// # Errors
///
/// Will return `Err` the window cannot be created / updated.
pub fn img(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    window_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let img_width = img.width() as usize;
    let img_height = img.height() as usize;
    let buffer = img.to_buffer();
    let mut window = Window::new(
        format!("{window_name} - Press Q or ESC to exit").as_str(),
        img_width,
        img_height,
        WindowOptions::default(),
    )?;

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        window.update_with_buffer(&buffer, img_width, img_height)?;
    }
    Ok(())
}

/// Visualize the given image and annotations.
///
/// ## Args
///
/// - `img_path`: The path to the image corresponding to the annotations.
/// - `anns`: The annotations to draw on the image. The annotations should all correspond to the same image.
/// - `draw_bbox`: If true, draw the bounding boxes.
///
/// ## Panics
///
/// Will panic if the image file cannot be read .
///
/// ## Errors
///
/// Will return `Err` if the COCO segmentation mask decompression fails.
pub fn anns(
    img_path: &PathBuf,
    anns: &Vec<&Annotation>,
    draw_option: DrawOption,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = utils::load_img(img_path)?;
    draw::anns(&mut img, anns, draw_option)?;
    self::img(
        &img,
        img_path
            .file_name()
            .map_or("Image", |file_name| file_name.to_str().unwrap_or("Image")),
    )?;
    Ok(())
}
