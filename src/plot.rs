#[cfg(feature = "plot")]
pub mod plotter {
    use plotters::prelude::*;

    /// Plot the graph of the given data.
    /// 'name' is the name of the file.
    /// 'x' is the x-axis data.
    /// 'ys' is the y-axis data.
    /// 'label' is the label of the data.
    /// 'path' is the path to save the graph.
    pub fn plot(
        name: &str,
        x: Vec<f64>,
        ys: Vec<Vec<f64>>,
        label: Vec<String>,
        path: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ys = ys.iter().map(|v| v.to_vec()).collect::<Vec<_>>();

        let width = 1280;
        let height = 720;
        let path = match path {
            Some(p) => format!("{}/{}.png", p, name),
            None => format!("./graph/{}.png", name),
        };
        let root = BitMapBackend::new(&path, (width, height)).into_drawing_area();

        root.fill(&WHITE)?;

        let (y_min, y_max) = ys.iter().fold((f64::NAN, f64::NAN), |(m, n), v| {
            (
                v.iter().fold(m, |m, n| n.min(m)),
                v.iter().fold(n, |m, n| n.max(m)),
            )
        });

        let font = ("sans-serif", 32);

        let mut chart;

        if y_min.is_sign_negative() {
            chart = ChartBuilder::on(&root)
                .caption(name, font.into_font())
                .margin(10)
                .x_label_area_size(16)
                .y_label_area_size(42)
                .build_cartesian_2d(
                    (*x.first().unwrap() - 0.1)..(*x.last().unwrap() + 0.1),
                    (y_min - 0.1)..(y_max + 0.1),
                )?;
        } else {
            chart = ChartBuilder::on(&root)
                .caption(name, font.into_font())
                .margin(10)
                .x_label_area_size(16)
                .y_label_area_size(42)
                .build_cartesian_2d(
                    (*x.first().unwrap() - 0.1)..(*x.last().unwrap() + 0.1),
                    0f64..(y_max + 0.1),
                )?;
        }

        chart.configure_mesh().draw()?;

        let mut line_series = vec![];

        let color = [
            RGBColor(0, 0, 255),
            RGBColor(255, 0, 0),
            RGBColor(0, 255, 0),
            RGBColor(255, 255, 0),
            RGBColor(255, 0, 255),
            RGBColor(0, 255, 255),
            RGBColor(128, 128, 128),
            RGBColor(128, 0, 0),
            RGBColor(128, 128, 0),
            RGBColor(0, 128, 0),
            RGBColor(128, 0, 128),
            RGBColor(0, 128, 128),
            RGBColor(0, 0, 128),
            RGBColor(0, 0, 0),
        ];

        for (i, (y, l)) in ys.iter().zip(label.iter()).enumerate() {
            let line =
                LineSeries::new(x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)), color[i % 14])
                    .point_size(4);
            line_series.push((line, l));
        }

        for (line, l) in line_series {
            chart.draw_series(line)?.label(l);
        }

        Ok(())
    }
}
