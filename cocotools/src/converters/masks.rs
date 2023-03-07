use image;
use imageproc::drawing;
use ndarray::{s, Array2, ArrayViewMut, ShapeBuilder};
use thiserror::Error;

use crate::annotations::coco;
use crate::argparse::Segmentation;

/// A boolean mask indicating for each pixel whether it belongs to the object or not.
pub type Mask = Array2<u8>;

/// # Errors
///
/// Will return `Err` if the conversion failed.
pub fn convert_coco_segmentation(
    dataset: &mut coco::HashmapDataset,
    target_segmentation: Segmentation,
) -> Result<(), MaskError> {
    let anns: Vec<coco::Annotation> = dataset.get_anns().into_iter().cloned().collect();
    for ann in anns {
        let converted_segmentation = match &ann.segmentation {
            coco::Segmentation::Rle(rle) => match target_segmentation {
                Segmentation::Rle => coco::Segmentation::Rle(rle.clone()),
                Segmentation::EncodedRle => {
                    coco::Segmentation::EncodedRle(coco::EncodedRle::try_from(rle)?)
                }
                Segmentation::Polygon => coco::Segmentation::Polygon(coco::Polygon::from(rle)),
            },
            coco::Segmentation::EncodedRle(_encoded_rle) => todo!(),
            coco::Segmentation::PolygonRS(poly) => match target_segmentation {
                Segmentation::Rle => coco::Segmentation::Rle(coco::Rle::from(poly)),
                Segmentation::EncodedRle => todo!(),
                Segmentation::Polygon => coco::Segmentation::Polygon(vec![poly.counts.clone()]),
            },
            coco::Segmentation::Polygon(_) => unimplemented!(),
        };
        dataset.add_ann(&coco::Annotation {
            segmentation: converted_segmentation,
            ..ann.clone()
        });
    }
    Ok(())
}

impl From<&coco::Rle> for coco::Polygon {
    fn from(_rle: &coco::Rle) -> Self {
        todo!()
    }
}

impl From<&coco::PolygonRS> for coco::Rle {
    // It might be more efficient to do it like this: https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c#L162
    // It would also avoid having slightly different results from the reference implementation.
    fn from(poly: &coco::PolygonRS) -> Self {
        coco::Rle::from(&Mask::from(poly))
    }
}

/// Decode encoded rle segmentation information into a rle.

/// See the (hard to read) implementation:
/// <https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c#L218>
/// <https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/_mask.pyx#L145>

/// [LEB128 wikipedia article](https://en.wikipedia.org/wiki/LEB128#Decode_signed_integer)
/// It is similar to LEB128, but here shift is incremented by 5 instead of 7 because the implementation uses
/// 6 bits per byte instead of 8. (no idea why, I guess it's more efficient for the COCO dataset?)
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
impl From<&coco::EncodedRle> for coco::Rle {
    /// Converts a compressed RLE to its uncompressed version.
    fn from(encoded_rle: &coco::EncodedRle) -> Self {
        assert!(
            encoded_rle.counts.is_ascii(),
            "Encoded RLE is not in valid ascii."
        );

        let bytes_rle = encoded_rle.counts.as_bytes();

        let mut current_count_idx: usize = 0;
        let mut current_byte_idx: usize = 0;
        let mut counts: Vec<u32> = vec![0; encoded_rle.counts.len()];
        while current_byte_idx < bytes_rle.len() {
            let mut continuous_pixels: i32 = 0;
            let mut shift = 0;
            let mut high_order_bit = 1;

            // When the high order bit of a byte becomes 0, we have decoded the integer and can move on to the next one.
            while high_order_bit != 0 {
                let byte = bytes_rle[current_byte_idx] - 48; // The encoding uses the ascii chars 48-111.

                // 0x1f is 31, i.e. 001111 --> Here we select the first four bits of the byte.
                continuous_pixels |= (i32::from(byte) & 31) << shift;
                // 0x20 is 32 as int, i.e. 2**5, i.e 010000 --> Here we select the fifth bit of the byte.
                high_order_bit = byte & 32;
                current_byte_idx += 1;
                shift += 5;
                // 0x10 is 16 as int, i.e. 1000
                if high_order_bit == 0 && (byte & 16 != 0) {
                    continuous_pixels |= !0 << shift;
                }
            }

            if current_count_idx > 2 {
                // My hypothesis as to what is happening here, is that most objects are going to be somewhat
                // 'vertically convex' (i.e. have only one continuous run per line).
                // In which case, the next 'row' of black/white pixels is going to be similar to the one preceding it.
                // Therefore, by having the continuous count of pixels be an offset of the one preceding it, we can have it be
                // a smaller int and therefore use less bits to encode it.
                continuous_pixels += counts[current_count_idx - 2] as i32;
            }
            counts[current_count_idx] = continuous_pixels as u32;
            current_count_idx += 1;
        }

        // TODO: Added the while loop to pass the tests, but it should not be there. Something is wrong somewhere else.
        while let Some(last) = counts.last() {
            if *last == 0 {
                counts.pop();
            } else {
                break;
            }
        }

        Self {
            size: encoded_rle.size.clone(),
            counts,
        }
    }
}

impl TryFrom<&coco::Rle> for coco::EncodedRle {
    type Error = MaskError;

    // Get compressed string representation of encoded mask.
    fn try_from(rle: &coco::Rle) -> Result<Self, Self::Error> {
        let mut high_order_bit: bool;
        let mut byte: u8;
        let mut encoded_counts: Vec<u8> = Vec::new();

        for i in 0..rle.counts.len() {
            let mut continuous_pixels = i64::from(rle.counts[i]);
            if i > 2 {
                continuous_pixels -= i64::from(rle.counts[i - 2]);
            }
            high_order_bit = true;
            while high_order_bit {
                byte = u8::try_from(continuous_pixels & 0x1f)
                    .map_err(|err| MaskError::IntConversion(err, continuous_pixels & 0x1f))?;
                continuous_pixels >>= 5;
                high_order_bit = if byte & 0x10 == 0 {
                    continuous_pixels != 0
                } else {
                    continuous_pixels != -1
                };
                if high_order_bit {
                    byte |= 0x20;
                };
                byte += 48;
                encoded_counts.push(byte);
            }
        }
        Ok(Self {
            size: rle.size.clone(),
            counts: std::str::from_utf8(&encoded_counts)
                .map_err(|err| MaskError::StrConversion(err, encoded_counts.clone()))?
                .to_string(),
        })
    }
}

impl From<&coco::Rle> for Mask {
    /// Converts a RLE to its uncompressed mask.
    fn from(rle: &coco::Rle) -> Self {
        let height = usize::try_from(rle.size[0]).unwrap();
        let width = usize::try_from(rle.size[1]).unwrap();

        let mut mask: Array2<u8> = Array2::zeros((height, width).f());
        let mut mask_1d = ArrayViewMut::from_shape(
            (height * width).f(),
            mask.as_slice_memory_order_mut().unwrap(),
        )
        .unwrap();

        let mut current_value = 0u8;
        let mut current_position = 0usize;
        for nb_pixels in &rle.counts {
            mask_1d
                .slice_mut(s![
                    current_position..current_position + usize::try_from(*nb_pixels).unwrap()
                ])
                .fill(current_value);
            current_value = u8::from(current_value == 0);
            current_position += usize::try_from(*nb_pixels).unwrap();
        }
        mask
    }
}

/// Convert a mask into its RLE form.
///
/// ## Args:
/// - mask: A binary mask indicating for each pixel whether it belongs to the object or not.
///
/// ## Returns:
/// - The RLE corresponding to the mask.
///
/// ## TODO: Find a way to avoid the .clone()  (if possible while still taking a reference)
impl From<&Mask> for coco::Rle {
    fn from(mask: &Mask) -> Self {
        let mut previous_value = 0;
        let mut count = 0;
        let mut counts = Vec::new();
        for value in mask.clone().reversed_axes().iter() {
            if *value != previous_value {
                counts.push(count);
                previous_value = *value;
                count = 0;
            }
            count += 1;
        }
        counts.push(count);

        Self {
            size: vec![mask.nrows() as u32, mask.ncols() as u32],
            counts,
        }
    }
}

impl From<&coco::Segmentation> for Mask {
    fn from(coco_segmentation: &coco::Segmentation) -> Self {
        match coco_segmentation {
            coco::Segmentation::Rle(rle) => Self::from(rle),
            coco::Segmentation::EncodedRle(encoded_rle) => {
                Self::from(&coco::Rle::from(encoded_rle))
            }
            coco::Segmentation::PolygonRS(poly) => Self::from(poly),
            coco::Segmentation::Polygon(_) => {
                unimplemented!("Use the 'mask_from_poly' function.")
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
impl From<&coco::PolygonRS> for Mask {
    /// Create a mask from a compressed polygon representation.
    fn from(poly: &coco::PolygonRS) -> Self {
        let mut points_poly: Vec<imageproc::point::Point<i32>> = Vec::new();
        for i in (0..poly.counts.len()).step_by(2) {
            points_poly.push(imageproc::point::Point::new(
                poly.counts[i] as i32,
                poly.counts[i + 1] as i32,
            ));
        }
        if let Some(last_point) = points_poly.last() {
            if points_poly[0].x == last_point.x && points_poly[0].y == last_point.y {
                points_poly.pop();
            }
        }

        let mut mask = image::GrayImage::new(poly.size[1], poly.size[0]);
        drawing::draw_polygon_mut(&mut mask, &points_poly, image::Luma([1u8]));

        Mask::from_shape_vec(
            (poly.size[1] as usize, poly.size[0] as usize),
            mask.into_raw(),
        )
        .unwrap()
    }
}

#[allow(clippy::cast_possible_truncation)]
pub fn mask_from_poly(poly: &coco::Polygon, width: u32, height: u32) -> Mask {
    let mut points_poly: Vec<imageproc::point::Point<i32>> = Vec::new();
    for i in (0..poly[0].len()).step_by(2) {
        points_poly.push(imageproc::point::Point::new(
            poly[0][i] as i32,
            poly[0][i + 1] as i32,
        ));
    }
    let mut mask = image::GrayImage::new(width, height);
    drawing::draw_polygon_mut(&mut mask, &points_poly, image::Luma([1u8]));

    Mask::from_shape_vec((height as usize, width as usize), mask.into_raw()).unwrap()
}

#[derive(Debug, Error)]
pub enum MaskError {
    #[error("Failed to convert RLE to its compressed version due to a type conversion error. Tried to convert '{1:?}' to u8 and failed.")]
    IntConversion(#[source] std::num::TryFromIntError, i64),
    #[error("Failed to convert RLE to its compressed version due to a type conversion error. Tried to convert '{1:?}' to u8 and failed.")]
    StrConversion(#[source] std::str::Utf8Error, Vec<u8>),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::coco::{EncodedRle, Rle};
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;
    use rstest::rstest;

    prop_compose! {
        #[allow(clippy::unwrap_used)]
        fn generate_rle(max_value: u32, max_elts: usize)
            (counts in prop::collection::vec(1..max_value, 2..max_elts))
            (width in 1..counts.iter().sum(), sum in Just(counts.iter().sum::<u32>()), mut counts in Just(counts))
             -> Rle {
                let height = sum / width + 1;
                *counts.last_mut().unwrap() += width * height - sum;
                Rle { counts,
                      size: vec![width, height]
                }
            }
    }

    prop_compose! {
        fn generate_mask(max_ncols: usize, max_nrows: usize)
            (ncols in 2..max_ncols, nrows in 2..max_nrows)
            (ncols in Just(ncols),
             nrows in Just(nrows),
             mask_data in prop::collection::vec(0..=1u8, (ncols * nrows) as usize),
            ) -> Mask {
                Mask::from_shape_vec((nrows, ncols), mask_data).unwrap()
            }
    }

    proptest! {
        #[test]
        fn rle_decode_inverts_encode(rle in generate_rle(50, 20)){
            let encoded_rle = EncodedRle::try_from(&rle).unwrap();
            let decoded_rle = Rle::from(&encoded_rle);
            prop_assert_eq!(decoded_rle, rle);
        }
    }

    proptest! {
        #[test]
        fn mask_to_rle_to_mask(mask in generate_mask(100, 100)){
            let rle = Rle::from(&mask);
            let decoded_mask = Mask::from(&rle);
            prop_assert_eq!(decoded_mask, mask);
        }
    }

    #[rstest]
    #[case::square(
        &array![[0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]],
        &Rle {size: vec![4, 4], counts: vec![5, 2, 2, 2, 5]})]
    #[case::horizontal_line(
        &array![[0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
        &Rle {size: vec![4, 5], counts: vec![1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2]})]
    #[case::vertical_line(
        &array![[0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]],
        &Rle {size: vec![4, 5], counts: vec![8, 4, 8]})]
    fn mask_to_rle(#[case] mask: &Mask, #[case] expected_rle: &Rle) {
        let rle = Rle::from(mask);
        assert_eq!(&rle, expected_rle);
    }

    #[rstest]
    #[case::square(
        &array![[0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]],
        &Rle {size: vec![4, 4], counts: vec![5, 2, 2, 2, 5]})]
    #[case::horizontal_line(
        &array![[0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]],
        &Rle {size: vec![4, 5], counts: vec![1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2]})]
    #[case::vertical_line(
        &array![[0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]],
        &Rle {size: vec![4, 5], counts: vec![8, 4, 8]})]
    fn rle_to_mask(#[case] expected_mask: &Mask, #[case] rle: &Rle) {
        let mask = Mask::from(rle);
        assert_eq!(&mask, expected_mask);
    }

    #[rstest]
    #[case::square(&Rle {counts: vec![6, 1, 40, 4, 5, 4, 5, 4, 21], size: vec![9, 10]},
                     &EncodedRle {size: vec![9, 10], counts: "61X13mN000`0".to_string()})]
    #[case::test1(&Rle {counts: vec![245, 5, 35, 5, 35, 5, 35, 5, 35, 5, 1190], size: vec![40, 40]},
                  &EncodedRle {size: vec![40, 40], counts: "e75S10000000ST1".to_string()})]
    fn encode_rle(#[case] rle: &Rle, #[case] expected_encoded_rle: &EncodedRle) {
        let encoded_rle = EncodedRle::try_from(rle).unwrap();
        assert_eq!(&encoded_rle, expected_encoded_rle);
    }
}
