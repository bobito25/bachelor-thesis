

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b a)
(on c f)
(on d c)
(on e d)
(on f b)
(clear e)
)
(:goal
(and
(on a c)
(on d a))
)
)


