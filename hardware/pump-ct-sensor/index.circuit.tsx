const xiaoPinLabels = {
  pin1: "D0_A0",
  pin2: "D1_A1",
  pin3: "D2_A2",
  pin4: "D3_A3",
  pin5: "D4_SDA",
  pin6: "D5_SCL",
  pin7: "D6_TX",
  pin8: "D7_RX",
  pin9: "D8_SCK",
  pin10: "D9_MISO",
  pin11: "D10_MOSI",
  pin12: "V3_3",
  pin13: "GND",
  pin14: "V5",
} as const

export default () => (
  <board width="70mm" height="50mm" layers={2}>
    {/* === XIAO ESP32-S3 Module === */}
    <chip
      name="U1"
      footprint="dip14_w17.78mm"
      pinLabels={xiaoPinLabels}
      schPinArrangement={{
        leftSide: {
          direction: "top-to-bottom",
          pins: ["D0_A0", "D1_A1", "D2_A2", "D3_A3", "D4_SDA", "D5_SCL", "D6_TX"],
        },
        rightSide: {
          direction: "top-to-bottom",
          pins: ["V5", "GND", "V3_3", "D10_MOSI", "D9_MISO", "D8_SCK", "D7_RX"],
        },
      }}
      pcbX={24}
      pcbY={0}
      schX={10}
      schY={-3}
    />

    {/* === CT Clamp Input via USB-C (used as 2-wire connector) === */}
    <chip
      name="J1"
      manufacturerPartNumber="TYPE-C-31-M-12"
      supplierPartNumbers={{ jlcpcb: ["C165948"] }}
      pinLabels={{
        1: ["GND1", "A1"],
        2: ["GND2", "B12"],
        3: ["VBUS1", "A4"],
        4: ["VBUS2", "B9"],
        5: ["SBU2", "B8"],
        6: ["CC1", "A5"],
        7: ["DM2", "B7"],
        8: ["DP1", "A6"],
        9: ["DM1", "A7"],
        10: ["DP2", "B6"],
        11: ["SBU1", "A8"],
        12: ["CC2", "B5"],
        13: ["VBUS3", "A9"],
        14: ["VBUS4", "B4"],
        15: ["GND3", "A12"],
        16: ["GND4", "B1"],
        17: ["ALT0", "alt_0"],
        18: ["ALT1", "alt_1"],
        19: ["ALT2", "alt_2"],
        20: ["ALT3", "alt_3"],
      }}
      schPinArrangement={{
        rightSide: {
          pins: ["VBUS1", "GND1"],
          direction: "top-to-bottom",
        },
      }}
      footprint={
        <footprint>
          <hole pcbX="-2.9mm" pcbY="1.18mm" diameter="0.75mm" />
          <hole pcbX="2.9mm" pcbY="1.18mm" diameter="0.75mm" />
          <platedhole portHints={["alt_2"]} pcbX="4.33mm" pcbY="-2.77mm" outerHeight="1.8mm" outerWidth="1.2mm" innerHeight="1.4mm" innerWidth="0.8mm" height="1.4mm" shape="pill" />
          <platedhole portHints={["alt_1"]} pcbX="4.33mm" pcbY="1.41mm" outerHeight="2mm" outerWidth="1.2mm" innerHeight="1.6mm" innerWidth="0.8mm" height="1.6mm" shape="pill" />
          <platedhole portHints={["alt_0"]} pcbX="-4.33mm" pcbY="1.41mm" outerHeight="2mm" outerWidth="1.2mm" innerHeight="1.6mm" innerWidth="0.8mm" height="1.6mm" shape="pill" />
          <platedhole portHints={["alt_3"]} pcbX="-4.33mm" pcbY="-2.77mm" outerHeight="1.8mm" outerWidth="1.2mm" innerHeight="1.4mm" innerWidth="0.8mm" height="1.4mm" shape="pill" />
          <smtpad portHints={["B8"]} pcbX="-1.75mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A5"]} pcbX="-1.25mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B7"]} pcbX="-0.75mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A6"]} pcbX="-0.25mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A7"]} pcbX="0.25mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B6"]} pcbX="0.75mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A8"]} pcbX="1.25mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B5"]} pcbX="1.75mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A1"]} pcbX="-3.35mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B12"]} pcbX="-3.05mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A4"]} pcbX="-2.55mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B9"]} pcbX="-2.25mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B4"]} pcbX="2.25mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A9"]} pcbX="2.55mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["B1"]} pcbX="3.05mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
          <smtpad portHints={["A12"]} pcbX="3.35mm" pcbY="2.45mm" width="0.3mm" height="1.3mm" shape="rect" />
        </footprint>
      }
      pcbX={-28}
      pcbY={0}
      schX={-12}
      schY={-3}
    />

    {/* Burden resistor across CT clamp (bridges CT+ to CT-) */}
    <resistor
      name="R1"
      resistance="56"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={-19}
      pcbY={0}
      schX={-9}
      schY={-3}
    />

    {/* === Bias Voltage Divider === */}
    <resistor
      name="R2"
      resistance="100k"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={-14}
      pcbY={12}
      schX={-6}
      schY={-5}
    />
    <resistor
      name="R3"
      resistance="100k"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={-14}
      pcbY={-4}
      schX={-6}
      schY={-1}
    />
    <capacitor
      name="C2"
      capacitance="1uF"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={-8}
      pcbY={-12}
      schX={-4}
      schY={-3}
    />

    {/* === RC Low-Pass Filter Stage 1 === */}
    <resistor
      name="R4"
      resistance="1k"
      footprint="axial_p5mm"
      schOrientation="horizontal"
      pcbX={2}
      pcbY={6}
      schX={0}
      schY={-3}
    />
    <capacitor
      name="C4"
      capacitance="100nF"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={2}
      pcbY={-6}
      schX={2}
      schY={-5}
    />

    {/* === RC Low-Pass Filter Stage 2 === */}
    <resistor
      name="R5"
      resistance="1k"
      footprint="axial_p5mm"
      schOrientation="horizontal"
      pcbX={10}
      pcbY={6}
      schX={4}
      schY={-3}
    />
    <capacitor
      name="C5"
      capacitance="100nF"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={10}
      pcbY={-6}
      schX={6}
      schY={-5}
    />

    {/* === Ferrite Bead for Analog Power Filtering === */}
    <inductor
      name="FB1"
      inductance="600ohm@100MHz"
      footprint="0805"
      schOrientation="horizontal"
      pcbX={18}
      pcbY={12}
      schX={-8}
      schY={-7}
    />

    {/* === Power Decoupling === */}
    <capacitor
      name="C6"
      capacitance="10uF"
      footprint="axial_p5mm"
      polarized
      schOrientation="vertical"
      pcbX={24}
      pcbY={18}
      schX={-5}
      schY={-9}
    />
    <capacitor
      name="C7"
      capacitance="100nF"
      footprint="axial_p5mm"
      schOrientation="vertical"
      pcbX={24}
      pcbY={-18}
      schX={-3}
      schY={-9}
    />

    {/* === Net Labels === */}
    <netlabel net="V3_3_ANALOG" schX={-4} schY={-7} anchorSide="bottom" />
    <netlabel net="VBIAS" schX={-6} schY={-3} anchorSide="right" />
    <netlabel net="ADC_RAW" schX={-2} schY={-3} anchorSide="top" />
    <netlabel net="NODE_A" schX={3} schY={-3} anchorSide="top" />
    <netlabel net="GPIO2" schX={7} schY={-3} anchorSide="top" />

    {/* === Schematic Boxes for Functional Groups === */}
    <schematicbox overlay={["FB1", "C6", "C7"]} padding="0.5mm" title="Power Filtering" strokeStyle="dashed" />
    <schematicbox overlay={["R2", "R3", "C2"]} padding="0.5mm" title="Bias Divider" strokeStyle="dashed" />
    <schematicbox overlay={["R4", "C4", "R5", "C5"]} padding="0.5mm" title="2-Stage RC Filter" strokeStyle="dashed" />

    {/* === Ferrite Bead Traces (V3_3 → V3_3_ANALOG) === */}
    <trace from=".FB1 > .pin1" to="net.V3_3" />
    <trace from=".FB1 > .pin2" to="net.V3_3_ANALOG" />

    {/* === Power Decoupling Traces (on filtered analog rail) === */}
    <trace from=".C6 > .pin1" to="net.V3_3_ANALOG" />
    <trace from=".C6 > .pin2" to="net.GND" />
    <trace from=".C7 > .pin1" to="net.V3_3_ANALOG" />
    <trace from=".C7 > .pin2" to="net.GND" />

    {/* === Bias Voltage Divider Traces === */}
    <trace from=".R2 > .pin1" to="net.V3_3_ANALOG" />
    <trace from=".R2 > .pin2" to="net.VBIAS" />
    <trace from=".R3 > .pin1" to="net.VBIAS" />
    <trace from=".R3 > .pin2" to="net.GND" />
    <trace from=".C2 > .pin1" to="net.VBIAS" />
    <trace from=".C2 > .pin2" to="net.GND" />

    {/* === CT Clamp Input Traces (USB-C as 2-wire: all VBUS→CT+, all GND→CT-) === */}
    <trace from=".J1 > .VBUS1" to="net.ADC_RAW" />
    <trace from=".J1 > .VBUS2" to="net.ADC_RAW" />
    <trace from=".J1 > .VBUS3" to="net.ADC_RAW" />
    <trace from=".J1 > .VBUS4" to="net.ADC_RAW" />
    <trace from=".J1 > .GND1" to="net.VBIAS" />
    <trace from=".J1 > .GND2" to="net.VBIAS" />
    <trace from=".J1 > .GND3" to="net.VBIAS" />
    <trace from=".J1 > .GND4" to="net.VBIAS" />
    <trace from=".R1 > .pin1" to="net.ADC_RAW" />
    <trace from=".R1 > .pin2" to="net.VBIAS" />

    {/* === RC Filter Traces === */}
    <trace from=".R4 > .pin1" to="net.ADC_RAW" />
    <trace from=".R4 > .pin2" to="net.NODE_A" />
    <trace from=".C4 > .pin1" to="net.NODE_A" />
    <trace from=".C4 > .pin2" to="net.GND" />
    <trace from=".R5 > .pin1" to="net.NODE_A" />
    <trace from=".R5 > .pin2" to="net.GPIO2" />
    <trace from=".C5 > .pin1" to="net.GPIO2" />
    <trace from=".C5 > .pin2" to="net.GND" />

    {/* === ESP32 Connections (GPIO2 = pin2 / D1_A1) === */}
    <trace from=".U1 > .D1_A1" to="net.GPIO2" />
    <trace from=".U1 > .V3_3" to="net.V3_3" />
    <trace from=".U1 > .GND" to="net.GND" />
  </board>
)
